from collections import Counter
from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st


APP_TITLE = "Wmyst Happiness Dashboard"

DATA_SOURCES = [
    (Path("private_data/realwmyst.xlsx"), "Private Excel file"),
    (Path("private_data/realwmyst.csv"), "Private CSV file"),
    (Path("data/sample_data.csv"), "Sample CSV file"),
]

COLUMN_ALIASES = {
    "unnamed: 0": "Date",
    "date": "Date",
    "day rating": "Rating",
    "rating": "Rating",
    "wmyst": "Smile",
    "smile": "Smile",
}

REQUIRED_COLUMNS = ["Date", "Rating", "Smile"]

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

STOP_WORDS = {
    "the",
    "and",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "up",
    "out",
    "as",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "she",
    "they",
    "them",
    "his",
    "her",
    "their",
    "went",
    "got",
    "had",
    "did",
    "day",
    "today",
    "really",
    "very",
    "just",
    "also",
    "then",
}


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=":)",
    layout="wide",
)

alt.data_transformers.disable_max_rows()


def read_data_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def add_calendar_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = df["Date"].dt.year.astype(int)
    df["Month Number"] = df["Date"].dt.month.astype(int)
    df["Month Name"] = pd.Categorical(
        df["Date"].dt.month_name(),
        categories=MONTH_NAMES,
        ordered=True,
    )
    df["Month Label"] = df["Date"].dt.strftime("%Y-%m")
    df["Month Start"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column: COLUMN_ALIASES.get(str(column).strip().lower(), column)
        for column in raw_df.columns
    }

    df = raw_df.rename(columns=rename_map)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]

    if missing_columns:
        expected = ", ".join(REQUIRED_COLUMNS)
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Missing required column(s): {missing}. Expected columns: {expected}."
        )

    df = df[REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Date", "Rating", "Smile"])

    df["Smile"] = df["Smile"].astype(str).str.strip()
    df = df[df["Smile"].ne("")]
    df = df.sort_values("Date").reset_index(drop=True)

    if df.empty:
        raise ValueError("The data file was found, but no usable rows were loaded.")

    return add_calendar_columns(df)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, str]:
    for path, source_name in DATA_SOURCES:
        if path.exists():
            return clean_data(read_data_file(path)), source_name

    expected_paths = ", ".join(str(path) for path, _ in DATA_SOURCES)
    raise FileNotFoundError(f"Could not find one of these files: {expected_paths}")


def get_common_words(text_series: pd.Series, top_n: int = 20) -> list[tuple[str, int]]:
    all_text = " ".join(text_series.astype(str)).lower()
    words = re.findall(r"\b[a-zA-Z']+\b", all_text)
    filtered_words = [
        word for word in words if word not in STOP_WORDS and len(word) > 2
    ]
    return Counter(filtered_words).most_common(top_n)


def format_number(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def month_names_to_numbers(month_names: list[str]) -> list[int]:
    return [MONTH_NAMES.index(month_name) + 1 for month_name in month_names]


def filter_data(
    df: pd.DataFrame,
    date_range: tuple | list | None = None,
    years: list[int] | None = None,
    months: list[str] | None = None,
) -> pd.DataFrame:
    filtered = df.copy()

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered["Date"].dt.date >= start_date)
            & (filtered["Date"].dt.date <= end_date)
        ]

    if years:
        filtered = filtered[filtered["Year"].isin(years)]

    if months:
        filtered = filtered[
            filtered["Month Number"].isin(month_names_to_numbers(months))
        ]

    return filtered


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Year",
        "Month Number",
        "Month Name",
        "Month Label",
        "Month Start",
        "Average Rating",
        "Days Tracked",
        "Highest Rating",
        "Lowest Rating",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        df.groupby(
            ["Year", "Month Number", "Month Name", "Month Label", "Month Start"],
            observed=True,
        )
        .agg(
            Average_Rating=("Rating", "mean"),
            Days_Tracked=("Rating", "size"),
            Highest_Rating=("Rating", "max"),
            Lowest_Rating=("Rating", "min"),
        )
        .reset_index()
        .sort_values("Month Start")
    )

    summary["Average Rating"] = summary["Average_Rating"].round(2)
    summary["Days Tracked"] = summary["Days_Tracked"].astype(int)
    summary["Highest Rating"] = summary["Highest_Rating"].round(2)
    summary["Lowest Rating"] = summary["Lowest_Rating"].round(2)
    summary["Month Name"] = summary["Month Name"].astype(str)

    return summary[
        [
            "Year",
            "Month Number",
            "Month Name",
            "Month Label",
            "Month Start",
            "Average Rating",
            "Days Tracked",
            "Highest Rating",
            "Lowest Rating",
        ]
    ]


def calendar_month_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Month Number", "Month Name", "Average Rating", "Days Tracked"]

    if df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        df.groupby(["Month Number", "Month Name"], observed=True)
        .agg(
            Average_Rating=("Rating", "mean"),
            Days_Tracked=("Rating", "size"),
        )
        .reset_index()
        .sort_values("Month Number")
    )

    summary["Average Rating"] = summary["Average_Rating"].round(2)
    summary["Days Tracked"] = summary["Days_Tracked"].astype(int)
    summary["Month Name"] = summary["Month Name"].astype(str)
    return summary[columns]


def month_by_year_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["Year", "Month Number", "Month Name", "Average Rating", "Days Tracked"]

    if df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        df.groupby(["Year", "Month Number", "Month Name"], observed=True)
        .agg(
            Average_Rating=("Rating", "mean"),
            Days_Tracked=("Rating", "size"),
        )
        .reset_index()
        .sort_values(["Year", "Month Number"])
    )

    summary["Average Rating"] = summary["Average_Rating"].round(2)
    summary["Days Tracked"] = summary["Days_Tracked"].astype(int)
    summary["Month Name"] = summary["Month Name"].astype(str)
    return summary[columns]


def render_rating_over_time_chart(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No ratings match the current filters.")
        return

    chart = (
        alt.Chart(df)
        .mark_line(point={"filled": True, "size": 35})
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Rating:Q", title="Rating", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%b %d, %Y"),
                alt.Tooltip("Rating:Q", title="Rating", format=".2f"),
                alt.Tooltip("Smile:N", title="Note"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)


def render_monthly_trend_chart(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No monthly averages are available for this selection.")
        return

    chart = (
        alt.Chart(summary)
        .mark_line(point={"filled": True, "size": 55})
        .encode(
            x=alt.X("Month Start:T", title="Month"),
            y=alt.Y(
                "Average Rating:Q",
                title="Average rating",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Month Label:N", title="Month"),
                alt.Tooltip("Average Rating:Q", title="Average", format=".2f"),
                alt.Tooltip("Days Tracked:Q", title="Days tracked"),
            ],
        )
        .properties(height=330)
    )

    st.altair_chart(chart, use_container_width=True)


def render_average_by_month_chart(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No calendar month averages are available for this selection.")
        return

    chart = (
        alt.Chart(summary)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("Month Name:N", title="Month", sort=MONTH_NAMES),
            y=alt.Y("Average Rating:Q", title="Average rating"),
            color=alt.Color(
                "Average Rating:Q",
                title="Average rating",
                scale=alt.Scale(scheme="tealblues"),
            ),
            tooltip=[
                alt.Tooltip("Month Name:N", title="Month"),
                alt.Tooltip("Average Rating:Q", title="Average", format=".2f"),
                alt.Tooltip("Days Tracked:Q", title="Days tracked"),
            ],
        )
        .properties(height=330)
    )

    st.altair_chart(chart, use_container_width=True)


def render_month_across_years_chart(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No year-by-month averages are available for this selection.")
        return

    chart = (
        alt.Chart(summary)
        .mark_line(point={"filled": True, "size": 55})
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y(
                "Average Rating:Q",
                title="Average rating",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color("Month Name:N", title="Month", sort=MONTH_NAMES),
            tooltip=[
                alt.Tooltip("Year:O", title="Year"),
                alt.Tooltip("Month Name:N", title="Month"),
                alt.Tooltip("Average Rating:Q", title="Average", format=".2f"),
                alt.Tooltip("Days Tracked:Q", title="Days tracked"),
            ],
        )
        .properties(height=360)
    )

    st.altair_chart(chart, use_container_width=True)


def render_month_year_heatmap(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("No month/year heatmap is available for this selection.")
        return

    chart = (
        alt.Chart(summary)
        .mark_rect()
        .encode(
            x=alt.X("Month Name:N", title="Month", sort=MONTH_NAMES),
            y=alt.Y("Year:O", title="Year", sort="-y"),
            color=alt.Color(
                "Average Rating:Q",
                title="Average rating",
                scale=alt.Scale(scheme="redyellowgreen"),
            ),
            tooltip=[
                alt.Tooltip("Year:O", title="Year"),
                alt.Tooltip("Month Name:N", title="Month"),
                alt.Tooltip("Average Rating:Q", title="Average", format=".2f"),
                alt.Tooltip("Days Tracked:Q", title="Days tracked"),
            ],
        )
        .properties(height=360)
    )

    st.altair_chart(chart, use_container_width=True)


def close_monthly_dashboard() -> None:
    st.session_state["show_monthly_dashboard"] = False


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def keep_multiselect_state_valid(key: str, options: list, default: list) -> None:
    if key not in st.session_state:
        return

    valid_values = [value for value in st.session_state[key] if value in options]
    st.session_state[key] = valid_values or default


def monthly_dashboard_content(df: pd.DataFrame) -> None:
    st.subheader("Monthly dashboard")

    if df.empty:
        st.warning("No data is available for the current filters.")
        return

    available_years = sorted(int(year) for year in df["Year"].unique())
    available_month_numbers = sorted(int(month) for month in df["Month Number"].unique())
    available_months = [MONTH_NAMES[month - 1] for month in available_month_numbers]

    keep_multiselect_state_valid(
        "monthly_dashboard_months",
        available_months,
        available_months,
    )
    keep_multiselect_state_valid(
        "monthly_dashboard_years",
        available_years,
        available_years,
    )

    filter_col_1, filter_col_2 = st.columns(2)

    with filter_col_1:
        selected_months = st.multiselect(
            "Months",
            options=available_months,
            default=available_months,
            key="monthly_dashboard_months",
        )

    with filter_col_2:
        selected_years = st.multiselect(
            "Years",
            options=available_years,
            default=available_years,
            key="monthly_dashboard_years",
        )

    dashboard_df = filter_data(df, years=selected_years, months=selected_months)

    if dashboard_df.empty:
        st.warning("No entries match the dashboard month and year selections.")
        return

    monthly = monthly_summary(dashboard_df)
    calendar_months = calendar_month_summary(dashboard_df)
    by_year = month_by_year_summary(dashboard_df)

    best_month = monthly.sort_values("Average Rating", ascending=False).iloc[0]
    best_calendar_month = calendar_months.sort_values(
        "Average Rating",
        ascending=False,
    ).iloc[0]

    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric(
        "Average of Months",
        format_number(monthly["Average Rating"].mean()),
    )
    metric_col_2.metric(
        "Best Month",
        f"{best_month['Month Label']} ({format_number(best_month['Average Rating'])})",
    )
    metric_col_3.metric(
        "Best Calendar Month",
        f"{best_calendar_month['Month Name']} ({format_number(best_calendar_month['Average Rating'])})",
    )
    metric_col_4.metric("Days Tracked", f"{len(dashboard_df):,}")

    trend_tab, averages_tab, years_tab, table_tab = st.tabs(
        ["Monthly trend", "Calendar averages", "Across years", "Tables"]
    )

    with trend_tab:
        render_monthly_trend_chart(monthly)

    with averages_tab:
        render_average_by_month_chart(calendar_months)

    with years_tab:
        chart_col_1, chart_col_2 = st.columns(2)
        with chart_col_1:
            render_month_across_years_chart(by_year)
        with chart_col_2:
            render_month_year_heatmap(by_year)

    with table_tab:
        st.dataframe(
            monthly[
                [
                    "Month Label",
                    "Average Rating",
                    "Days Tracked",
                    "Highest Rating",
                    "Lowest Rating",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            calendar_months,
            use_container_width=True,
            hide_index=True,
        )

    if st.button("Close dashboard", key="close_monthly_dashboard_button"):
        close_monthly_dashboard()
        rerun_app()


def get_dialog_decorator(title: str):
    dialog = getattr(st, "dialog", None)

    if dialog is not None:
        for kwargs in (
            {"width": "large", "on_dismiss": close_monthly_dashboard},
            {"width": "large"},
            {},
        ):
            try:
                return dialog(title, **kwargs)
            except TypeError:
                continue

    experimental_dialog = getattr(st, "experimental_dialog", None)

    if experimental_dialog is not None:
        try:
            return experimental_dialog(title)
        except TypeError:
            return None

    return None


def _monthly_dashboard_modal(df: pd.DataFrame) -> None:
    monthly_dashboard_content(df)


dialog_decorator = get_dialog_decorator("Monthly Dashboard")
HAS_DIALOG = dialog_decorator is not None

if HAS_DIALOG:
    monthly_dashboard_modal = dialog_decorator(_monthly_dashboard_modal)
else:
    monthly_dashboard_modal = _monthly_dashboard_modal


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    year_options = sorted(int(year) for year in df["Year"].unique())
    month_options = [MONTH_NAMES[int(month) - 1] for month in sorted(df["Month Number"].unique())]

    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    selected_years = st.sidebar.multiselect(
        "Years",
        options=year_options,
        default=year_options,
    )

    selected_months = st.sidebar.multiselect(
        "Months",
        options=month_options,
        default=month_options,
    )

    return filter_data(
        df,
        date_range=date_range,
        years=selected_years,
        months=selected_months,
    )


def render_overview_metrics(df: pd.DataFrame) -> None:
    col_1, col_2, col_3, col_4 = st.columns(4)
    col_1.metric("Average Rating", format_number(df["Rating"].mean()))
    col_2.metric("Highest Rating", format_number(df["Rating"].max()))
    col_3.metric("Lowest Rating", format_number(df["Rating"].min()))
    col_4.metric("Days Tracked", f"{len(df):,}")


def render_common_words(df: pd.DataFrame) -> None:
    st.subheader("Most Common Words")
    common_words = get_common_words(df["Smile"], top_n=20)

    if not common_words:
        st.info("No common words are available for this selection.")
        return

    words_df = pd.DataFrame(common_words, columns=["Word", "Count"])
    chart_col, table_col = st.columns([2, 1])

    with chart_col:
        chart = (
            alt.Chart(words_df)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("Count:Q", title="Count"),
                y=alt.Y("Word:N", title=None, sort="-x"),
                tooltip=["Word:N", "Count:Q"],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)

    with table_col:
        st.dataframe(words_df, use_container_width=True, hide_index=True)


def render_lowest_periods(df: pd.DataFrame) -> None:
    st.subheader("Lowest Rated 7-Day Periods")

    rolling_df = df.set_index("Date").copy()
    rolling_df["7 Day Average"] = rolling_df["Rating"].rolling("7D").mean()

    lowest_periods = (
        rolling_df.dropna(subset=["7 Day Average"])
        .reset_index()
        .sort_values("7 Day Average")
        .head(10)
    )

    if lowest_periods.empty:
        st.info("There is not enough data for a 7-day rolling average.")
        return

    lowest_periods["7 Day Average"] = lowest_periods["7 Day Average"].round(2)

    st.dataframe(
        lowest_periods[["Date", "Rating", "7 Day Average", "Smile"]],
        use_container_width=True,
        hide_index=True,
    )


def render_recent_entries(df: pd.DataFrame) -> None:
    st.subheader("Recent Entries")
    recent_entries = df.sort_values("Date", ascending=False).head(20)
    st.dataframe(
        recent_entries[["Date", "Rating", "Smile", "Month Label"]],
        use_container_width=True,
        hide_index=True,
    )


def render_main_app() -> None:
    try:
        df, source = load_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    st.title(APP_TITLE)
    st.caption(
        f"Loaded from: {source} | {len(df):,} entries from "
        f"{df['Date'].min().date()} to {df['Date'].max().date()}"
    )

    filtered_df = render_sidebar_filters(df)

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    st.subheader("Overview")
    render_overview_metrics(filtered_df)

    if st.button("Open monthly dashboard"):
        st.session_state["show_monthly_dashboard"] = True

    if st.session_state.get("show_monthly_dashboard", False):
        if HAS_DIALOG:
            monthly_dashboard_modal(filtered_df)
        else:
            with st.expander("Monthly Dashboard", expanded=True):
                monthly_dashboard_modal(filtered_df)

    st.subheader("Rating Over Time")
    render_rating_over_time_chart(filtered_df)

    st.subheader("Best Monthly Averages")
    best_months = monthly_summary(filtered_df).sort_values(
        "Average Rating",
        ascending=False,
    )
    st.dataframe(
        best_months[
            [
                "Month Label",
                "Average Rating",
                "Days Tracked",
                "Highest Rating",
                "Lowest Rating",
            ]
        ].head(10),
        use_container_width=True,
        hide_index=True,
    )

    render_common_words(filtered_df)
    render_lowest_periods(filtered_df)
    render_recent_entries(filtered_df)


render_main_app()
