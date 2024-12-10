import pandas as pd

# Feature name mapping dictionary
feature_name_mapping = {
    # Primary Focus Subject Mappings
    "onehotencoder__primary_focus_subject_Mathematics": "Primary Focus: Mathematics",
    "onehotencoder__primary_focus_subject_Literature & Writing": "Primary Focus: Literature & Writing",
    "onehotencoder__primary_focus_subject_College & Career Prep": "Primary Focus: College & Career Prep",
    "onehotencoder__primary_focus_subject_Parent Involvement": "Primary Focus: Parent Involvement",
    "onehotencoder__primary_focus_subject_Literacy": "Primary Focus: Literacy",

    # Poverty Level
    "ordinalencoder__poverty_level": "Poverty Level",

    # Resource Type Mappings
    "onehotencoder__resource_type_Technology": "Resource Type: Technology",
    "onehotencoder__resource_type_Supplies": "Resource Type: Supplies",
    "onehotencoder__resource_type_Books": "Resource Type: Books",
    "onehotencoder__resource_type_Trips": "Resource Type: Trips",
    "onehotencoder__resource_type_Visitors": "Resource Type: Visitors",
    "onehotencoder__resource_type_Other": "Resource Type: Other",

    # Grade Level Mappings
    "onehotencoder__grade_level_Grades 3-5": "Grade Level: Grades 3-5",
    "onehotencoder__grade_level_Grades 6-8": "Grade Level: Grades 6-8",
    "onehotencoder__grade_level_Grades 9-12": "Grade Level: Grades 9-12",
    "onehotencoder__grade_level_Grades PreK-2": "Grade Level: Grades PK-2",

    # School State Mappings (examples for some states)
    "onehotencoder__school_state_CA": "School State: California",
    "onehotencoder__school_state_TX": "School State: Texas",
    "onehotencoder__school_state_NY": "School State: New York",
    "onehotencoder__school_state_FL": "School State: Florida",
    "onehotencoder__school_state_IL": "School State: Illinois",
    "onehotencoder__school_state_GA": "School State: Georgia",
    "onehotencoder__school_state_NC": "School State: North Carolina",
    "onehotencoder__school_state_MA": "School State: Massachusetts",
    "onehotencoder__school_state_AZ": "School State: Arizona",
    "onehotencoder__school_state_PA": "School State: Pennsylvania",
    "onehotencoder__school_state_OH": "School State: Ohio",
    "onehotencoder__school_state_MI": "School State: Michigan",
    "onehotencoder__school_state_WA": "School State: Washington",
    "onehotencoder__school_state_CO": "School State: Colorado",
    "onehotencoder__school_state_SC": "School State: South Carolina",
    "onehotencoder__school_state_MD": "School State: Maryland",
    "onehotencoder__school_state_NJ": "School State: New Jersey",
    "onehotencoder__school_state_IN": "School State: Indiana",
    "onehotencoder__school_state_AL": "School State: Alabama",
    "onehotencoder__school_state_LA": "School State: Louisiana",
    "onehotencoder__school_state_OR": "School State: Oregon",
    "onehotencoder__school_state_VA": "School State: Virginia",
    "onehotencoder__school_state_NM": "School State: New Mexico",
    "onehotencoder__school_state_UT": "School State: Utah",
    "onehotencoder__school_state_CT": "School State: Connecticut",
    "onehotencoder__school_state_KS": "School State: Kansas",
    "onehotencoder__school_state_MO": "School State: Missouri",
    "onehotencoder__school_state_NV": "School State: Nevada",
    "onehotencoder__school_state_ME": "School State: Maine",
    "onehotencoder__school_state_HI": "School State: Hawaii",
    "onehotencoder__school_state_ID": "School State: Idaho",
    "onehotencoder__school_state_WV": "School State: West Virginia",
    "onehotencoder__school_state_NE": "School State: Nebraska",
    "onehotencoder__school_state_VT": "School State: Vermont",
    "onehotencoder__school_state_SD": "School State: South Dakota",
    "onehotencoder__school_state_WY": "School State: Wyoming",
    "onehotencoder__school_state_ND": "School State: North Dakota",
    "onehotencoder__school_state_MN": "School State: Minnesota",
    "onehotencoder__school_state_KY": "School State: Kentucky",
    "onehotencoder__school_state_OK": "School State: Oklahoma",
    "onehotencoder__school_state_IA": "School State: Iowa",
    "onehotencoder__school_state_AR": "School State: Arkansas",
    "onehotencoder__school_state_MS": "School State: Mississippi",
    "onehotencoder__school_state_WI": "School State: Wisconsin",
    "onehotencoder__school_state_MT": "School State: Montana",
    "onehotencoder__school_state_TN": "School State: Tennessee",
    "onehotencoder__school_state_AK": "School State: Alaska",
    "onehotencoder__school_state_NH": "School State: New Hampshire",
    "onehotencoder__school_state_DE": "School State: Delaware",
    "onehotencoder__school_state_RI": "School State: Rhode Island",
    "onehotencoder__school_state_DC": "School State: District of Columbia",

    # School Characteristics
    "remainder__school_charter": "School Type: Charter",
    "remainder__school_magnet": "School Type: Magnet",
    "remainder__school_year_round": "School Year: Year-Round",
    "remainder__school_nlns": "School Type: NLNS",
    "remainder__school_kipp": "School Type: KIPP",
    "remainder__school_charter_ready_promise": "School Type: Charter Ready Promise",

    # Fulfillment & Cost
    "remainder__fulfillment_labor_materials": "Cost: Fulfillment Labor & Materials",

    # Essay and Description Counts
    "remainder__essay_word_count": "Essay Word Count",
    "remainder__description_word_count": "Description Word Count",

    # Other Feature Mappings
    "remainder__students_reached": "Number of Students Reached",
    "remainder__teacher_prefix": "Teacher Prefix",
    "remainder__primary_focus_area": "Primary Focus Area",
    "remainder__secondary_focus_area": "Secondary Focus Area",
    "remainder__need_word_count": "Need Word Count",
    "remainder__title_word_count": "Title Word Count",
    "remainder__sub_primary_focus_area": "Sub-Primary Focus Area",
    "onehotencoder__primary_focus_subject_Literacy": "Primary Focus: Literacy",
    "onehotencoder__primary_focus_subject_Parent Involvement": "Primary Focus: Parent Involvement",
    "onehotencoder__primary_focus_subject_Other": "Primary Focus: Other",
}

def apply_feature_name_mapping(importance_df):
    """
    Maps feature names to more interpretable labels.

    Parameters:
    - importance_df: DataFrame containing feature importances with a column 'Feature'.

    Returns:
    - A DataFrame with updated feature names.
    """
    importance_df['Feature'] = importance_df['Feature'].apply(
        lambda x: feature_name_mapping.get(x, x)
    )
    return importance_df

if __name__ == "__main__":
    # Example usage
    data = {'Feature': ['onehotencoder__resource_type_Books', 'onehotencoder__school_state_CA']}
    df = pd.DataFrame(data)
    updated_df = apply_feature_name_mapping(df)
    print(updated_df)
