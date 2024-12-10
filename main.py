from src.clean.cleaning_general import clean_data as clean_general
from src.clean.cleaning_rural import clean_data as clean_rural
from src.models.log_reg import train_logistic_regression
from src.models.xgboost import train_xgboost
from src.models.grad_boost import train_gradient_boosting
from src.models.naives_bayes import train_naive_bayes
from src.models.catboost import train_catboost
from src.utils.baseline import run_baseline
from src.evaluation.feature_importance import plot_feature_importance
from src.evaluation.prk_curve import plot_prk_curve
from src.tuning.naive_bayes_tuning import tune_naive_bayes
from src.tuning.xgboost_tuning import tune_xgboost
from src.analysis.eda import plot_correlation_heatmap, plot_numerical_distributions, plot_categorical_counts

def main(args):
    # Load data
    data_file = args.data
    print(f"Using data file: {data_file}")

    # Step 1: Exploratory Data Analysis (EDA)
    if args.eda:
        print("\nRunning Exploratory Data Analysis...")
        data = train_validation_test_split(data_file)[0]  # Load data for EDA
        numerical_cols = ['students_reached', 'fulfillment_labor_materials', 'title_word_count',
                          'description_word_count', 'need_word_count', 'essay_word_count']
        readable_names = {
            'students_reached': 'Students Reached',
            'fulfillment_labor_materials': 'Fulfillment Cost',
            'title_word_count': 'Title Word Count',
            'description_word_count': 'Description Word Count',
            'need_word_count': 'Need Word Count',
            'essay_word_count': 'Essay Word Count'
        }
        categorical_cols = ['resource_type', 'school_state', 'grade_level']

        plot_correlation_heatmap(data, readable_names)
        plot_numerical_distributions(data, numerical_cols)
        plot_categorical_counts(data, categorical_cols)
        plot_specific_distributions(data)

    # Step 2: Run Baseline
    if args.baseline:
        print("\nRunning Baseline...")
        run_baseline(data_file)

    # Step 3: Train Models
    if args.models:
        print("\nTraining Models...")
        train_logistic_regression_lasso(data_file)
        train_xgboost(data_file)
        train_gradient_boosting(data_file)
        train_naive_bayes(data_file)
        train_catboost(data_file)

    # Step 4: Hyperparameter Tuning
    if args.tuning:
        print("\nRunning Hyperparameter Tuning...")
        tune_xgboost(data_file)
        tune_naive_bayes(data_file)

    # Step 5: Evaluation
    if args.evaluation:
        print("\nEvaluating Models...")
        # Example: Naive Bayes Feature Importance
        nb_model = train_naive_bayes(data_file, return_model=True)
        feature_names = train_validation_test_split(data_file)[6]  # Get feature names
        plot_feature_importance(nb_model, feature_names)

        # Example: PR-K Curve for Naive Bayes
        X_test, y_test = train_validation_test_split(data_file)[2:4]  # Get test data
        plot_prk_curve(nb_model, X_test, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Machine Learning Pipeline")
    parser.add_argument("--data", type=str, default="data/data_imp_rural.csv", help="Path to the data file.")
    parser.add_argument("--eda", action="store_true", help="Run Exploratory Data Analysis (EDA).")
    parser.add_argument("--baseline", action="store_true", help="Run the baseline model.")
    parser.add_argument("--models", action="store_true", help="Train all models.")
    parser.add_argument("--tuning", action="store_true", help="Run hyperparameter tuning.")
    parser.add_argument("--evaluation", action="store_true", help="Run evaluation and plotting.")

    args = parser.parse_args()
    main(args)