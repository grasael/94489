import os
import pandas as pd

def convert_to_binary(value):
    """
    Convert "t" to 1 and "f" to 0. Returns the original value if neither.
    """
    return 1 if value == "t" else 0 if value == "f" else value

def clean_general_data(projects_file, outcomes_file, essays_file, output_file):
    """
    Cleans and processes the input datasets, merging them into a single cleaned file.
    
    Parameters:
    - projects_file: Path to the projects dataset CSV.
    - outcomes_file: Path to the outcomes dataset CSV.
    - essays_file: Path to the essays dataset CSV.
    - output_file: Path to save the cleaned dataset CSV.
    """
    # Check if files exist
    for file in [projects_file, outcomes_file, essays_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file not found: {file}. Please download it to the appropriate location.")

    # Load datasets
    projects = pd.read_csv(projects_file)
    outcomes = pd.read_csv(outcomes_file)
    essays = pd.read_csv(essays_file)
    
    # Process outcomes
    outcomes["not_fully_funded"] = outcomes["fully_funded"].apply(lambda x: 0 if x == "t" else 1)
    outcomes = outcomes.drop(columns=["fully_funded"])

    # Select relevant project fields
    projects = projects[[
        "projectid", "primary_focus_subject", "poverty_level",
        "students_reached", "resource_type", "fulfillment_labor_materials",
        "school_state", "school_charter", "school_magnet",
        "school_year_round", "school_nlns", "school_kipp",
        "school_charter_ready_promise", "grade_level"
    ]]

    # Convert boolean columns to binary
    boolean_columns = ["school_charter", "school_magnet", "school_year_round", 
                       "school_nlns", "school_kipp", "school_charter_ready_promise"]
    for column in boolean_columns:
        projects[column] = projects[column].apply(convert_to_binary)

    # Merge datasets
    data = (
        projects
        .merge(outcomes, on="projectid")
        .merge(essays, on="projectid")
        .drop(columns=["projectid", "teacher_acctid"])
        .dropna()
    )

    # Filter important projects
    def important_project(row):
        important_subjects = [
            "Mathematics", "Literature & Writing", "College & Career Prep", 
            "Parent Involvement", "Literacy"
        ]
        important_poverty = ["highest poverty", "high poverty"]
        return "yes" if row["primary_focus_subject"] in important_subjects and row["poverty_level"] in important_poverty else "no"

    data["important"] = data.apply(important_project, axis=1)
    data_imp = data[data["important"] == "yes"].drop(columns=["important", "primary_focus_subject", "poverty_level"])

    # Save cleaned data
    data_imp.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    clean_general_data(
        "dataset/projects.csv",
        "dataset/outcomes.csv",
        "dataset/essays_info.csv",
        "data/data_imp.csv"
    )
