"""Streamlit tab for the Fasthouse SKU checker.

Maintains the plain-text SKU list the checker Lambda reads from S3 and
lets it be run on demand. The EventBridge schedule (Mon/Fri) is not
touched - this only changes how the list gets updated and adds a manual
run.
"""
import logging
from datetime import time as dt_time
from typing import List, Optional

import pandas as pd
import streamlit as st

from crawler_app.awsio import (
    AwsConfig,
    SkuCheckerClient,
    detect_newline,
    diff_skus,
    format_skus,
    merge_skus,
    parse_skus,
)
from crawler_app.jobs import REGISTRY
from crawler_app.schedule import (
    DAY_LABELS,
    DAY_NAMES,
    Schedule,
    ScheduleClient,
    build_weekly_cron,
    describe_expression,
    next_runs,
    parse_weekly_cron,
)

log = logging.getLogger(__name__)

SCHEDULE_POLICY = """{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadAndEditTheSkuCheckerSchedule",
      "Effect": "Allow",
      "Action": [
        "events:ListRuleNamesByTarget",
        "events:DescribeRule",
        "events:PutRule",
        "events:EnableRule",
        "events:DisableRule",
        "scheduler:ListSchedules",
        "scheduler:GetSchedule",
        "scheduler:UpdateSchedule"
      ],
      "Resource": "*"
    }
  ]
}"""

SKU_COLUMNS = ("Seller SKU", "SKU", "Seller-SKU", "seller sku")


def load_config() -> AwsConfig:
    """Read the [aws] block from Streamlit secrets; missing file is fine."""
    try:
        raw = dict(st.secrets.get("aws", {}))
    except Exception:  # noqa: BLE001 - no secrets.toml at all
        raw = {}
    return AwsConfig(
        region=str(raw.get("region", "")).strip(),
        bucket=str(raw.get("bucket", "")).strip(),
        sku_key=str(raw.get("sku_key", "")).strip(),
        lambda_arn=str(raw.get("lambda_arn", "")).strip(),
        access_key_id=str(raw.get("access_key_id", "")).strip(),
        secret_access_key=str(raw.get("secret_access_key", "")).strip(),
        sku_prefix=str(raw.get("sku_prefix", "")).strip(),
        backup_prefix=str(raw.get("backup_prefix", "")).strip(),
        results_prefix=str(raw.get("results_prefix", "")).strip(),
        lambda_payload=dict(raw.get("lambda_payload", {}) or {}),
    )


@st.cache_resource(show_spinner=False)
def _client(region: str, bucket: str, key_id: str, secret: str, sku_key: str, lambda_arn: str,
            sku_prefix: str, backup_prefix: str, results_prefix: str) -> SkuCheckerClient:
    return SkuCheckerClient(AwsConfig(
        region=region, bucket=bucket, access_key_id=key_id, secret_access_key=secret,
        sku_key=sku_key, lambda_arn=lambda_arn, sku_prefix=sku_prefix,
        backup_prefix=backup_prefix, results_prefix=results_prefix,
    ))


def get_client(config: AwsConfig) -> SkuCheckerClient:
    return _client(config.region, config.bucket, config.access_key_id, config.secret_access_key,
                   config.sku_key, config.lambda_arn, config.sku_prefix,
                   config.backup_prefix, config.results_prefix)


def skus_from_dataframe(df: pd.DataFrame) -> List[str]:
    for column in SKU_COLUMNS:
        match = next((c for c in df.columns if str(c).strip().lower() == column.lower()), None)
        if match:
            values = df[match].astype(str).str.strip()
            return parse_skus("\n".join(v for v in values if v and v.lower() != "nan"))
    return []


def _setup_help() -> None:
    st.info("The SKU checker is not configured yet.", icon="ℹ️")
    st.markdown(
        """
Add an `[aws]` block to **`.streamlit/secrets.toml`** locally, and to
*Settings → Secrets* on Streamlit Cloud:

```toml
[aws]
region            = "us-east-1"
bucket            = "permanent-data-bucket"
sku_key           = "inputs/fasthouse/SKUS.txt"   # exact key of the SKU list
sku_prefix        = "inputs/"                     # used to browse for it
lambda_arn        = "arn:aws:lambda:us-east-1:065614340143:function:FH_New_SKU_Checker"
results_prefix    = "sku-available/"              # optional: where results land
access_key_id     = "AKIA..."
secret_access_key = "..."
```

The IAM user needs `s3:GetObject` / `s3:PutObject` on the SKU key and its
`backups/` folder, `s3:ListBucket` on the bucket, and
`lambda:InvokeFunction` on the function.
        """
    )


def _render_current_list(client: SkuCheckerClient, key: str):
    """Show what the Lambda will read; returns the parsed current list."""
    try:
        text, meta = client.read_text(key)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Could not read `{key}`: {e}")
        return None

    current = parse_skus(text)
    st.session_state["sku_newline"] = detect_newline(text)
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs in the list", f"{len(current):,}")
    c2.metric("Last updated", meta.last_modified or "unknown")
    c3.metric("Size", f"{meta.size:,} bytes")
    with st.expander(f"View current list ({len(current):,} SKUs)", expanded=False):
        st.code("\n".join(current) or "(empty)", language="text")
    return current


def _source_skus(config: AwsConfig) -> List[str]:
    """Let the user choose where the new SKU list comes from."""
    job = REGISTRY.current
    options = ["Upload a file", "Paste SKUs"]
    if job is not None and job.result_df is not None:
        options.insert(0, "Last crawl result")

    source = st.radio("Where should the SKUs come from?", options, horizontal=True)

    if source == "Last crawl result":
        skus = skus_from_dataframe(job.result_df)
        st.caption(f"{len(skus):,} SKUs from the {job.website} / {job.crawl_type} run finished at "
                   f"{pd.to_datetime(job.finished_at, unit='s').strftime('%H:%M') if job.finished_at else 'n/a'}.")
        return skus

    if source == "Upload a file":
        uploaded = st.file_uploader("CSV with a 'Seller SKU' column, or a .txt with one SKU per line",
                                    type=["csv", "txt"], key="sku_source_file")
        if uploaded is None:
            return []
        raw = uploaded.getvalue()
        if uploaded.name.lower().endswith(".csv"):
            try:
                import io
                return skus_from_dataframe(pd.read_csv(io.BytesIO(raw)))
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the CSV: {e}")
                return []
        return parse_skus(raw.decode("utf-8", errors="replace"))

    pasted = st.text_area("One SKU per line", height=160, key="sku_source_paste")
    return parse_skus(pasted)


def _render_update(client: SkuCheckerClient, key: str, current: Optional[List[str]], config: AwsConfig) -> None:
    st.markdown("#### Update the SKU list")
    new_skus = _source_skus(config)
    if not new_skus:
        st.caption("Choose a source above to see what would change.")
        return

    mode = st.radio(
        "How should the list be updated?",
        ["Add new SKUs to the existing list", "Replace the list entirely"],
        help="Adding keeps every SKU already in the file; replacing writes only the SKUs from the source.",
    )
    merging = mode.startswith("Add")
    base = current or []
    final = merge_skus(base, new_skus) if merging else list(new_skus)
    delta = diff_skus(base, final)

    c1, c2, c3 = st.columns(3)
    c1.metric("Added", f"{len(delta['added']):,}")
    c2.metric("Removed", f"{len(delta['removed']):,}", delta=f"-{len(delta['removed'])}" if delta["removed"] else None,
              delta_color="inverse")
    c3.metric("List after update", f"{len(final):,}")

    if delta["added"]:
        with st.expander(f"{len(delta['added']):,} SKU(s) to be added", expanded=True):
            st.code("\n".join(delta["added"]), language="text")
    if delta["removed"]:
        with st.expander(f"⚠️ {len(delta['removed']):,} SKU(s) that would be REMOVED", expanded=True):
            st.code("\n".join(delta["removed"]), language="text")

    if not delta["added"] and not delta["removed"]:
        st.success("The list in S3 already matches this source - nothing to write.")
        return

    ending = st.session_state.get("sku_newline", "\n")
    st.download_button("⬇ Preview the file that would be written", data=format_skus(final, ending),
                       file_name=key.rsplit("/", 1)[-1], mime="text/plain", on_click="ignore")

    if st.button("💾 Write this list to S3", type="primary"):
        try:
            with st.spinner("Writing to S3..."):
                result = client.update_sku_list(key, final, backup=True, newline=ending)
        except Exception as e:  # noqa: BLE001
            st.error(f"Update failed: {e}")
            return
        st.success(f"Wrote {result['count']:,} SKUs to `{result['key']}`.")
        if result.get("backup_key"):
            st.caption(f"Previous version kept at `{result['backup_key']}`.")


def _render_run(client: SkuCheckerClient, config: AwsConfig) -> None:
    st.markdown("#### Run the checker now")
    st.caption(f"Invokes **{config.function_name}** immediately. The Monday/Friday schedule is unaffected.")

    if st.button("▶ Run SKU checker", type="primary", disabled=not config.can_invoke):
        try:
            with st.spinner("Running the Lambda - this can take a while for a long list..."):
                result = client.invoke_checker()
        except Exception as e:  # noqa: BLE001
            st.error(f"Invoke failed: {e}")
            return
        st.session_state["sku_check_result"] = result

    result = st.session_state.get("sku_check_result")
    if not result:
        return
    if result.get("function_error"):
        st.error(f"The function reported an error ({result['function_error']}) after {result['elapsed']:.0f}s.")
    elif result.get("status_code") == 200:
        st.success(f"Finished in {result['elapsed']:.0f}s (HTTP {result['status_code']}).")
    else:
        st.warning(f"HTTP {result.get('status_code')} after {result['elapsed']:.0f}s.")
    if result.get("payload"):
        with st.expander("Lambda response", expanded=bool(result.get("function_error"))):
            st.code(result["payload"][:20000], language="json")


def _render_results(client: SkuCheckerClient, config: AwsConfig) -> None:
    if not config.results_prefix:
        return
    st.markdown("#### Latest results")
    try:
        objects = client.list_objects(config.results_prefix, limit=200)
    except Exception as e:  # noqa: BLE001
        st.caption(f"Could not list `{config.results_prefix}`: {e}")
        return
    if not objects:
        st.caption(f"No files under `{config.results_prefix}` yet.")
        return
    objects.sort(key=lambda o: (o.last_modified or "", o.key), reverse=True)
    latest = objects[:10]
    chosen = st.selectbox("Result file", latest,
                          format_func=lambda o: f"{o.key}  ({o.last_modified}, {o.size:,} bytes)")
    if chosen and st.button("Open selected result"):
        try:
            text, _ = client.read_text(chosen.key)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read it: {e}")
            return
        st.download_button("⬇ Download", data=text, file_name=chosen.key.rsplit("/", 1)[-1],
                           on_click="ignore")
        st.code(text[:20000], language="text")


# ---------------------------------------------------------------- schedule

@st.cache_resource(show_spinner=False)
def _schedule_client(region: str, lambda_arn: str, key_id: str, secret: str) -> ScheduleClient:
    return ScheduleClient(region=region, lambda_arn=lambda_arn,
                          access_key_id=key_id, secret_access_key=secret)


def get_schedule_client(config: AwsConfig) -> ScheduleClient:
    return _schedule_client(config.region, config.lambda_arn,
                            config.access_key_id, config.secret_access_key)


def _schedule_summary(sched: Schedule) -> str:
    state = "enabled" if sched.enabled else "DISABLED"
    return f"{describe_expression(sched.expression, sched.timezone)} - {state}"


def _render_schedule(config: AwsConfig) -> None:
    st.markdown("#### Schedule")
    st.caption("When AWS runs the checker on its own. Changing this edits the live "
               "EventBridge schedule; the manual run above is unaffected.")

    client = get_schedule_client(config)

    if st.button("Load schedule", key="load_schedule") or "schedules" not in st.session_state:
        with st.spinner("Looking for schedules that target the function..."):
            try:
                st.session_state["schedules"] = client.find_schedules()
            except Exception as e:  # noqa: BLE001
                st.session_state["schedules"] = []
                st.error(f"Could not read schedules: {e}")

    schedules = st.session_state.get("schedules") or []
    if not schedules:
        st.info("No schedule found for this function. Either the IAM user cannot read "
                "EventBridge yet, or the Monday/Friday run is set up another way.")
        with st.expander("Permissions needed to manage the schedule"):
            st.code(SCHEDULE_POLICY, language="json")
        return

    sched = schedules[0]
    if len(schedules) > 1:
        sched = st.selectbox("Schedule", schedules,
                             format_func=lambda s: f"{s.name} ({s.service}) - {_schedule_summary(s)}")

    service_label = "EventBridge rule" if sched.service == "rule" else "EventBridge Scheduler"
    st.markdown(f"**{sched.name}** · {service_label}")
    c1, c2 = st.columns([2, 1])
    c1.metric("Runs", describe_expression(sched.expression, sched.timezone))
    c2.metric("State", "Enabled" if sched.enabled else "Disabled")
    st.caption(f"Expression: `{sched.expression}`")

    upcoming = next_runs(sched.expression, sched.timezone, count=4)
    if upcoming and sched.enabled:
        st.caption("Next runs: " + ", ".join(r.strftime("%a %d %b %H:%M") for r in upcoming)
                   + f" ({sched.timezone})")

    # ---- editor ----
    parsed = parse_weekly_cron(sched.expression)
    if parsed is None:
        st.warning("This schedule uses an expression this app does not edit "
                   f"(`{sched.expression}`). Change it in the AWS console to avoid breaking it.")
    else:
        current_days, hour, minute = parsed
        with st.form("schedule_form"):
            days = st.multiselect("Days", DAY_NAMES, default=current_days,
                                  format_func=lambda d: DAY_LABELS[d])
            col_a, col_b = st.columns(2)
            new_time = col_a.time_input("Time", value=dt_time(hour=hour, minute=minute), step=300)
            if sched.supports_timezone:
                timezone = col_b.text_input("Timezone", value=sched.timezone,
                                            help="IANA name, e.g. America/Los_Angeles or UTC.")
            else:
                timezone = "UTC"
                col_b.text_input("Timezone", value="UTC", disabled=True,
                                 help="EventBridge rules always run in UTC.")
            confirm = st.checkbox("I understand this changes the live schedule")
            submitted = st.form_submit_button("Save schedule", type="primary")

        if submitted:
            try:
                expression = build_weekly_cron(days, new_time.hour, new_time.minute)
            except ValueError as e:
                st.error(str(e))
                return
            if expression == sched.expression and timezone == sched.timezone:
                st.info("That is already the schedule - nothing to change.")
                return
            if not confirm:
                st.warning("Tick the confirmation box to save.")
                return
            try:
                with st.spinner("Updating the schedule..."):
                    updated = client.update_expression(sched, expression, timezone)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not update the schedule: {e}")
                return
            st.session_state["schedules"] = [updated] + [s for s in schedules if s.name != updated.name]
            st.success(f"Now runs {describe_expression(updated.expression, updated.timezone)}.")
            preview = next_runs(updated.expression, updated.timezone, count=3)
            if preview:
                st.caption("Next runs: " + ", ".join(r.strftime("%a %d %b %H:%M") for r in preview))

    # ---- pause / resume ----
    label = "Pause the schedule" if sched.enabled else "Resume the schedule"
    if st.button(label, key="toggle_schedule"):
        try:
            with st.spinner("Updating..."):
                updated = client.set_enabled(sched, not sched.enabled)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not change the state: {e}")
            return
        st.session_state["schedules"] = [updated] + [s for s in schedules if s.name != updated.name]
        st.success("Schedule " + ("enabled." if updated.enabled else "paused - it will not run on its own."))


def render() -> None:
    """Draw the whole SKU checker tab."""
    st.subheader("Fasthouse SKU checker")

    config = load_config()
    if not config.has_credentials:
        _setup_help()
        return

    st.caption(f"Bucket `{config.bucket}` · region `{config.region}`"
               + (f" · function `{config.function_name}`" if config.lambda_arn else ""))

    client = get_client(config)

    with st.expander("Connection check", expanded=False):
        if st.button("Test connection"):
            with st.spinner("Checking..."):
                for label, ok, detail in client.check_access():
                    (st.success if ok else st.error)(f"{label} - {detail}")

    # Which file the Lambda reads. Configured key wins; otherwise browse.
    key = config.sku_key
    if not key:
        st.warning("No `sku_key` in secrets - pick the SKU list below and add it to secrets once confirmed.")
        try:
            candidates = client.list_objects(config.sku_prefix, suffix=".txt", limit=100)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not list `{config.sku_prefix}`: {e}")
            return
        if not candidates:
            st.error(f"No .txt files found under `{config.sku_prefix or '/'}`.")
            return
        picked = st.selectbox("SKU list file", candidates,
                              format_func=lambda o: f"{o.key}  ({o.last_modified}, {o.size:,} bytes)")
        key = picked.key if picked else ""
    if not key:
        return

    st.markdown(f"**SKU list:** `{key}`")
    current = _render_current_list(client, key)

    st.divider()
    _render_update(client, key, current, config)

    st.divider()
    _render_run(client, config)

    st.divider()
    _render_schedule(config)

    _render_results(client, config)
