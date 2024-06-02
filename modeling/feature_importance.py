from db_connection import DBConnection
from encoding import binary_encoding
import mlflow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #path = "mlflow-artifacts:/142205264287169719/a309cce1d995493294d94fbac8a4cee4/artifacts/best_model" #rf_t
    path = "mlflow-artifacts:/389766634362966321/28b143938f59488a9b2d6d2f6da1090a/artifacts/best_model" #xg_bb
    mlflow.set_tracking_uri("http://localhost:5000")
    model = mlflow.sklearn.load_model(path)
    rf = model["model"]
    feature_importance = rf.feature_importances_.tolist()

    db_connection = DBConnection()
    df = db_connection.get_dataframe()
    #feature_names = df.columns.drop("CustoIndustrial")

    df_encoded = binary_encoding(df)
    feature_names = df_encoded.columns.drop("CustoIndustrial")

    features_with_importance = zip(feature_names, feature_importance)

    sorted_features = sorted(features_with_importance, key=lambda x: x[1], reverse=True)
    
    feature_importance_list = []
    for feature_name, importance in sorted_features:
        feature_importance_list.append((feature_name, importance))


    #Plot

    threshold = 0.05
    high_importance_features = [(name, importance) for name, importance in sorted_features if importance > threshold]
    low_importance_features = [(name, importance) for name, importance in sorted_features if importance <= threshold]

    others_importance = sum(importance for name, importance in low_importance_features)
    
    high_importance_features.append(("Others", others_importance))

    # Create a bar plot
    graph_features, graph_importances = zip(*high_importance_features)

    color = '#1F77B4'

    plt.style.use('ggplot')    
    plt.figure(figsize=(10, 6))
    plt.barh(graph_features, graph_importances, color = color)
    plt.xlabel("Importance")
    #plt.ylabel("Feature")
    plt.title("Feature's Importance (xg_bb)")
    plt.gca().invert_yaxis()  # Invert y-axis to show the highest importance at the top
    plt.subplots_adjust(left=0.3)
    #plt.savefig("nome.png")

    from IPython import embed
    embed()
