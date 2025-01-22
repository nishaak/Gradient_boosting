from flask import Flask, render_template, request, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, classification_report, confusion_matrix, roc_curve, auc)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler 
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_EXTENSIONS'] = ['.csv', '.xlsx', '.tsv']
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
df = None
model = None

def label_encode(df, target_column):
    """Function to apply Label Encoding to categorical columns excluding the target column."""
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        if col != target_column:
            df[col] = label_encoder.fit_transform(df[col])
    return df

def one_hot_encode(df, target_column):
    """Function to apply One-Hot Encoding to categorical columns excluding the target column."""
    return pd.get_dummies(df, columns=[col for col in df.select_dtypes(include=['object']).columns if col != target_column], drop_first=True)


@app.route("/", methods=["GET", "POST"])
def main():
    global df, model
    accuracy, precision, recall, report, plot_url = None, None, None, None, None
    confusion_matrix_url, auc_roc_url = None, None
    columns = []
    column_info_html = None
    data_head = None
    missing_values_report = None
    summary_statistics = None
    eda_plots = []
    statistical_inferences = None

    if request.method == "POST":
        if "file" in request.files:
            file = request.files["file"]
            session['filename'] = file.filename
            file_extension = file.filename.split('.')[-1].lower()

            if f".{file_extension}" not in app.config['UPLOAD_EXTENSIONS']:
                flash("Invalid file type. Please upload a CSV, Excel, or TSV file.")
                return render_template("index.html", columns=columns, filename="No file chosen", data_head=data_head)

            try:
                if file_extension == "csv":
                    df = pd.read_csv(file)
                elif file_extension == "xlsx":
                    df = pd.read_excel(file, engine='openpyxl')
                elif file_extension == "tsv":
                    df = pd.read_csv(file, sep='\t')

                os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], "uploaded_data.csv"), index=False)

                columns = df.columns.tolist()
                df.replace('?', np.nan, inplace=True)
                session['columns'] = columns

                data_head = df.head(10).to_html(classes='data', header="true", index="false")

            except Exception as e:
                flash(f"Error reading file: {e}")
                return render_template("index.html", columns=columns, filename="No file chosen", data_head=data_head)
                session['columns'] = []
                
        elif request.form.get("action") == "Show Column Info" and df is not None:
            try:
                column_info = pd.DataFrame({
                     "Column Name": df.columns,
                     "Data Type": df.dtypes.astype(str)
                       })
                column_info_html = column_info.to_html(classes='data', header="true", index="false")
                flash("Column names and data types displayed successfully.")
            except Exception as e:
                flash(f"Error displaying column info: {e}")


        # Column Removal
        elif request.form.get("action") == "Remove Columns" and df is not None:
            selected_column = request.form.get("selected_column")  # Get the selected column
            if selected_column:
                try:
                    df.drop(columns=[selected_column], inplace=True)
                    session['columns'] = df.columns.tolist()  # Update columns in session
                    flash(f"Column '{selected_column}' removed successfully.")
                except Exception as e:
                    flash(f"Error removing column '{selected_column}': {e}")


        # Handle Exploratory Data Analysis
        elif request.form.get("action") == "Show Full Summary Statistics" and df is not None:
            try:
                summary_statistics = df.describe().to_html(classes='data', header="true", index="true")
                flash("Summary statistics for the whole dataset generated successfully.")
            except Exception as e:
                flash(f"Error generating summary statistics: {e}")

        elif request.form.get("action") == "Apply All" and df is not None:
            target_column = request.form.get("target_column")  # Get the selected target column
            fill_method = request.form.get("fill_method")
            encoding_method = request.form.get("encoding_method")
            scaling_method = request.form.get("scaling_method")

            try:
                columns_to_process = [col for col in df.columns if col != target_column]
                if fill_method == "Mean":
                    imputer = SimpleImputer(strategy='mean')
                    df[columns_to_process] = pd.DataFrame(
                        imputer.fit_transform(df[columns_to_process]),
                        columns=columns_to_process
            )
                    flash("Missing values handled using Mean, excluding the target column.")
                elif fill_method == "Median":
                    imputer = SimpleImputer(strategy='median')
                    df[columns_to_process] = pd.DataFrame(
                       imputer.fit_transform(df[columns_to_process]),
                       columns=columns_to_process
            )
                    flash("Missing values handled using Median, excluding the target column.")
                elif fill_method == "Mode":
                    imputer = SimpleImputer(strategy='most_frequent')
                    df[columns_to_process] = pd.DataFrame(
                        imputer.fit_transform(df[columns_to_process]),
                        columns=columns_to_process
            )
                    flash("Missing values handled using Mode, excluding the target column.")
                elif fill_method == "Drop":
                    df.dropna(subset=columns_to_process, inplace=True)
                    flash("Rows with missing values dropped, excluding the target column.")

        # Apply Encoding
                if encoding_method == "label":
                    df = label_encode(df, target_column)
                    flash("Categorical columns encoded using Label Encoding, excluding the target column.")
                elif encoding_method == "onehot":
                    df = one_hot_encode(df, target_column)
                    flash("Categorical columns encoded using One-Hot Encoding, excluding the target column.")

        # Apply Scaling
                numeric_data = df.select_dtypes(include=["number"]).drop(columns=[target_column])
                if scaling_method == "normalize":
                    scaler = MinMaxScaler()
                    df[numeric_data.columns] = pd.DataFrame(
                        scaler.fit_transform(numeric_data),
                        columns=numeric_data.columns
            )
                    flash("Data normalized using MinMaxScaler, excluding the target column.")
                elif scaling_method == "standardize":
                    scaler = StandardScaler()
                    df[numeric_data.columns] = pd.DataFrame(
                        scaler.fit_transform(numeric_data),
                        columns=numeric_data.columns
            )
                    flash("Data standardized using StandardScaler, excluding the target column.")
                elif scaling_method == "robust":
                    scaler = RobustScaler()
                    df[numeric_data.columns] = pd.DataFrame(
                        scaler.fit_transform(numeric_data),
                        columns=numeric_data.columns
            )
                    flash("Data scaled using RobustScaler, excluding the target column.")
                elif scaling_method == "none":
                    flash("No scaling applied.")

        # Update data preview
                df_preview = df.drop(columns=[target_column])
                data_head = df_preview.head(10).to_html(classes="data", header="true", index="false")
            except Exception as e:
                flash(f"Error during preprocessing: {e}")




        elif request.form.get("action") == "Generate Plot" and df is not None:
            x_axis = request.form.get("x_axis")
            y_axis = request.form.get("y_axis")
            plot_type = request.form.get("plot_type")

            if x_axis and (y_axis or plot_type == "missingness") and plot_type:
                plt.figure(figsize=(10, 6))
                try:
                    if plot_type == "scatter":
                        sns.scatterplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "line":
                        sns.lineplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "bar":
                        sns.barplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "hist":
                        sns.histplot(data=df[x_axis], kde=True)
                    elif plot_type == "violin":
                        sns.violinplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "swarm":
                        sns.swarmplot(data=df, x=x_axis, y=y_axis)
                    elif plot_type == "missingness":
                        import missingno as msno
                        msno.matrix(df)

                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    plot_url = base64.b64encode(buf.getvalue()).decode("utf8")
                    buf.close()
                    plt.close()

                except Exception as e:
                    flash(f"Error generating plot: {e}")


        elif request.form.get("action") == "Train Model" and df is not None:
            target_column = request.form.get("target")  # Get the target column from the form
            learning_rate = float(request.form.get("learning_rate", 0.1))  # Default to 0.1 if not specified
            n_estimators = int(request.form.get("n_estimators", 100))  # Default to 100 if not specified
            max_depth = int(request.form.get("max_depth", 3))  # Default to 3 if not specified
            optimization_method = request.form.get("optimization_method", "none")

            try:
                # Ensure the target column exists
                if target_column not in df.columns:
                    flash("Invalid target column selected.")
                    return render_template("index.html", columns=session.get('columns', []), data_head=None)

                # Separate features and target
                X = df.drop(columns=[target_column])
                y = df[target_column]

                # Apply label encoding to categorical features
                X = label_encode(X, target_column)

                # Split dataset into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Handle class imbalance with SMOTE
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

                # Initialize the Gradient Boosting model
                model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

                # Perform optimization if specified
                if optimization_method == "grid_search":
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.7, 0.8, 0.9]
                    }
                    search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=2)
                    search.fit(X_resampled, y_resampled)
                    model = search.best_estimator_
                elif optimization_method == "random_search":
                    param_grid = {
                        'n_estimators': [50, 100, 150],
                        'max_depth': [3, 5, 7],
                        'subsample': [0.7, 0.8, 0.9]
                    }
                    search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5, scoring='accuracy', random_state=42, verbose=2)
                    search.fit(X_resampled, y_resampled)
                    model = search.best_estimator_
                else:
                    model.fit(X_resampled, y_resampled)

                # Make predictions
                y_pred = model.predict(X_test)

                # Evaluate the model
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                report = classification_report(y_test, y_pred)

                # Generate confusion matrix plot
                conf_matrix = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                confusion_matrix_url = base64.b64encode(buf.getvalue()).decode("utf8")
                buf.close()
                plt.close()

                # Generate AUC-ROC curve (for binary classification)
                if y_test.nunique() == 2:  # Check for binary classification
                    y_prob = model.predict_proba(X_test)
                    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.title("AUC-ROC Curve")
                    plt.legend(loc="lower right")
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)
                    auc_roc_url = base64.b64encode(buf.getvalue()).decode("utf8")
                    buf.close()
                    plt.close()

            except Exception as e:
                flash(f"Error during model training: {e}")


        # Generate Statistical Inferences based on results
        if accuracy or precision or recall:
            statistical_inferences = f"""
            The trained model achieved an accuracy of {accuracy:.2f}, 
            a precision of {precision:.2f}, and a recall of {recall:.2f}.
            This suggests the model performs well on this dataset, 
            balancing between identifying true positives and avoiding false positives.
            """

        if summary_statistics:
            statistical_inferences = (statistical_inferences or "") + """
            The summary statistics indicate the central tendency and dispersion of the selected columns.
            Consider features with high variability for further analysis or normalization.
            """

        if eda_plots:
            statistical_inferences = (statistical_inferences or "") + """
            Distribution plots highlight the data's spread and potential outliers.
            Check for skewness or multimodal distributions that may require transformations.
            """

    columns = session.get('columns', [])
    filename = session.get('filename', 'No file chosen')


    return render_template("index.html", columns=columns, filename=filename, column_info_html=column_info_html,accuracy=accuracy, precision=precision, recall=recall, report=report, plot_url=plot_url, data_head=data_head, missing_values_report=missing_values_report, summary_statistics=summary_statistics, confusion_matrix_url=confusion_matrix_url, auc_roc_url=auc_roc_url, statistical_inferences=statistical_inferences)

if __name__ == "_main_":
    app.run(debug=True)