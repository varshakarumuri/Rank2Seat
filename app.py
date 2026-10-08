import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Rank2Seat",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# LOAD ML MODEL
# =========================================================

model = joblib.load(
    "model/best_rank2seat_model.pkl"
)


# =========================================================
# LOAD DATASET
# =========================================================

data = pd.read_csv(
    "dataset/2022.csv"
)

# Remove spaces from column names
data.columns = data.columns.str.strip()


# =========================================================
# CLEAN RANK COLUMNS
# =========================================================

data["Opening Rank"] = pd.to_numeric(
    data["Opening Rank"],
    errors="coerce"
)

data["Closing Rank"] = pd.to_numeric(
    data["Closing Rank"],
    errors="coerce"
)


# Remove rows with invalid rank values
data = data.dropna(
    subset=[
        "Opening Rank",
        "Closing Rank"
    ]
)


# Convert ranks to integers
data["Opening Rank"] = data[
    "Opening Rank"
].astype(int)

data["Closing Rank"] = data[
    "Closing Rank"
].astype(int)


# =========================================================
# PAGE TITLE
# =========================================================

st.title("🎓 Rank2Seat")

st.subheader(
    "JEE College & Seat Prediction System"
)

st.write(
    "Enter your JEE details to estimate "
    "your seat chances and find recommended "
    "colleges and branches."
)

st.divider()


# =========================================================
# STUDENT JEE RANK
# =========================================================

student_rank = st.number_input(
    "Enter your JEE Rank",
    min_value=1,
    value=10000,
    step=1
)


# =========================================================
# INSTITUTE
# =========================================================

institutes = sorted(
    data["Institute"]
    .dropna()
    .unique()
    .tolist()
)

institute = st.selectbox(
    "Select Institute",
    institutes
)


# =========================================================
# ACADEMIC PROGRAM / BRANCH
# =========================================================

program_data = data[
    data["Institute"] == institute
]

programs = sorted(
    program_data[
        "Academic Program Name"
    ]
    .dropna()
    .unique()
    .tolist()
)

program = st.selectbox(
    "Select Academic Program / Branch",
    programs
)


# =========================================================
# FILTER BY INSTITUTE + PROGRAM
# =========================================================

selected_data = data[
    (data["Institute"] == institute) &
    (
        data["Academic Program Name"]
        == program
    )
]


# =========================================================
# QUOTA
# =========================================================

quotas = sorted(
    selected_data["Quota"]
    .dropna()
    .unique()
    .tolist()
)

quota = st.selectbox(
    "Select Quota",
    quotas
)


# =========================================================
# FILTER BY QUOTA
# =========================================================

quota_data = selected_data[
    selected_data["Quota"] == quota
]


# =========================================================
# SEAT TYPE
# =========================================================

seat_types = sorted(
    quota_data["Seat Type"]
    .dropna()
    .unique()
    .tolist()
)

seat_type = st.selectbox(
    "Select Seat Type",
    seat_types
)


# =========================================================
# FILTER BY SEAT TYPE
# =========================================================

seat_data = quota_data[
    quota_data["Seat Type"] == seat_type
]


# =========================================================
# GENDER
# =========================================================

genders = sorted(
    seat_data["Gender"]
    .dropna()
    .unique()
    .tolist()
)

gender = st.selectbox(
    "Select Gender",
    genders
)


# =========================================================
# FILTER BY GENDER
# =========================================================

gender_data = seat_data[
    seat_data["Gender"] == gender
]


# =========================================================
# COUNSELLING ROUND
# =========================================================

rounds = sorted(
    gender_data["Round"]
    .dropna()
    .unique()
    .tolist()
)

round_number = st.selectbox(
    "Select Counselling Round",
    rounds
)


# =========================================================
# PREDICT BUTTON
# =========================================================

if st.button(
    "🔮 Predict",
    use_container_width=True
):

    # =====================================================
    # CREATE INPUT FOR ML MODEL
    # =====================================================

    input_data = pd.DataFrame([
        {
            "Institute": institute,
            "Academic Program Name": program,
            "Quota": quota,
            "Seat Type": seat_type,
            "Gender": gender,
            "Round": round_number
        }
    ])


    # =====================================================
    # ML PREDICTION
    # =====================================================

    predicted_closing_rank = model.predict(
        input_data
    )[0]

    predicted_closing_rank = round(
        predicted_closing_rank
    )


    # =====================================================
    # SELECTED COLLEGE PREDICTION
    # =====================================================

    st.divider()

    st.header(
        "📊 Selected College Prediction"
    )


    st.write(
        "**Your JEE Rank:**",
        student_rank
    )

    st.write(
        "**Institute:**",
        institute
    )

    st.write(
        "**Academic Program:**",
        program
    )

    st.write(
        "**Quota:**",
        quota
    )

    st.write(
        "**Seat Type:**",
        seat_type
    )

    st.write(
        "**Gender:**",
        gender
    )

    st.write(
        "**Counselling Round:**",
        round_number
    )

    st.write(
        "**Predicted Closing Rank:**",
        predicted_closing_rank
    )


    # =====================================================
    # SELECTED COLLEGE CHANCE
    # =====================================================

    if student_rank <= predicted_closing_rank:

        st.success(
            "🟢 GOOD CHANCE"
        )

    elif (
        student_rank
        <= predicted_closing_rank * 1.10
    ):

        st.warning(
            "🟡 MODERATE CHANCE"
        )

    else:

        st.error(
            "🔴 LOW CHANCE"
        )


    # =====================================================
    # RECOMMENDED COLLEGES & BRANCHES
    # =====================================================

    st.divider()

    st.header(
        "🎓 Recommended Colleges & Branches"
    )

    st.write(
        "Recommendations are based on your JEE rank "
        "and the selected Quota, Seat Type, Gender "
        "and Counselling Round."
    )


    # =====================================================
    # FILTER ALL COLLEGES
    # =====================================================

    recommendations = data[
        (data["Quota"] == quota) &
        (data["Seat Type"] == seat_type) &
        (data["Gender"] == gender) &
        (data["Round"] == round_number)
    ].copy()


    # =====================================================
    # CALCULATE CHANCE
    # =====================================================

    def calculate_chance(closing_rank):

        if student_rank <= closing_rank:

            return "🟢 GOOD"

        elif (
            student_rank
            <= closing_rank * 1.10
        ):

            return "🟡 MODERATE"

        else:

            return "🔴 LOW"


    recommendations["Chance"] = (
        recommendations[
            "Closing Rank"
        ].apply(calculate_chance)
    )


    # =====================================================
    # KEEP GOOD + MODERATE OPTIONS
    # =====================================================

    recommendations = recommendations[
        recommendations["Closing Rank"]
        >= student_rank * 0.90
    ]


    # =====================================================
    # SORT BY CLOSING RANK
    # =====================================================

    recommendations = (
        recommendations
        .sort_values(
            by="Closing Rank",
            ascending=True
        )
    )


    # =====================================================
    # REMOVE DUPLICATE COLLEGE + BRANCH ROWS
    # =====================================================

    recommendations = recommendations.drop_duplicates(
        subset=[
            "Institute",
            "Academic Program Name"
        ]
    )


    # =====================================================
    # DISPLAY RECOMMENDATIONS
    # =====================================================

    if recommendations.empty:

        st.warning(
            "No recommended colleges found "
            "for the selected options."
        )

    else:

        st.success(
            f"🎉 {len(recommendations)} "
            "college/branch options found!"
        )


        # =================================================
        # PREPARE DISPLAY DATA
        # =================================================

        display_data = recommendations[
            [
                "Institute",
                "Academic Program Name",
                "Opening Rank",
                "Closing Rank",
                "Chance"
            ]
        ].copy()


        # =================================================
        # DISPLAY TABLE
        # =================================================

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )