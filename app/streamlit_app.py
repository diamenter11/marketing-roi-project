# import streamlit as st
# import pandas as pd
# import joblib
# from pathlib import Path
# import sys

# sys.path.append(str(Path(__file__).resolve().parents[1]))

# from src.logger import get_logger


# logger = get_logger("streamlit_dashboard")

# DATA_PATH = Path("data/marketing_and_sales.csv")
# MODEL_PATH = Path("models/best_model.pkl")
# COMPARISON_PATH = Path("reports/model_comparison.csv")
# IMPORTANCE_PATH = Path("reports/feature_importance.csv")


# st.set_page_config(
#     page_title="Marketing ROI Optimizer",
#     page_icon="📊",
#     layout="wide"
# )


# @st.cache_data
# def load_data():
#     logger.info("Chargement du dataset dans Streamlit")

#     if not DATA_PATH.exists():
#         logger.error(f"Dataset introuvable : {DATA_PATH}")
#         raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")

#     return pd.read_csv(DATA_PATH)


# @st.cache_resource
# def load_model():
#     logger.info("Chargement du modèle dans Streamlit")

#     if not MODEL_PATH.exists():
#         logger.error(f"Modèle introuvable : {MODEL_PATH}")
#         raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

#     return joblib.load(MODEL_PATH)


# def safe_load_csv(path: Path):
#     if path.exists():
#         logger.info(f"Chargement du fichier : {path}")
#         return pd.read_csv(path)

#     logger.warning(f"Fichier optionnel absent : {path}")
#     return None


# def main():
#     st.title("📊 Marketing ROI Optimizer")
#     st.write(
#         "Dashboard décisionnel pour prédire les ventes "
#         "à partir d’un scénario budgétaire marketing."
#     )

#     try:
#         df = load_data()
#         model = load_model()
#     except Exception as error:
#         logger.exception("Erreur critique au démarrage du dashboard")
#         st.error(f"Erreur au chargement : {error}")
#         st.stop()

#     logger.info("Dashboard lancé avec succès")

#     st.sidebar.header("🎛️ Simulation budgétaire")

#     tv = st.sidebar.slider(
#         "Budget TV",
#         float(df["TV"].min()),
#         float(df["TV"].max()),
#         float(df["TV"].mean())
#     )

#     radio = st.sidebar.slider(
#         "Budget Radio",
#         float(df["Radio"].min()),
#         float(df["Radio"].max()),
#         float(df["Radio"].mean())
#     )

#     social_media = st.sidebar.slider(
#         "Budget Social Media",
#         float(df["Social Media"].min()),
#         float(df["Social Media"].max()),
#         float(df["Social Media"].mean())
#     )

#     influencer = st.sidebar.selectbox(
#         "Type d'influenceur",
#         sorted(df["Influencer"].unique())
#     )

#     input_data = pd.DataFrame([{
#         "TV": tv,
#         "Radio": radio,
#         "Social Media": social_media,
#         "Influencer": influencer
#     }])

#     try:
#         prediction = model.predict(input_data)[0]

#         logger.info(
#             "Prédiction effectuée | "
#             f"TV={tv:.2f}, Radio={radio:.2f}, "
#             f"Social Media={social_media:.2f}, Influencer={influencer}, "
#             f"Prediction={prediction:.2f}"
#         )

#     except Exception as error:
#         logger.exception("Erreur pendant la prédiction")
#         st.error(f"Erreur pendant la prédiction : {error}")
#         st.stop()

#     total_budget = tv + radio + social_media
#     roi = prediction / total_budget if total_budget > 0 else 0

#     col1, col2, col3 = st.columns(3)

#     col1.metric("💰 Budget total", f"{total_budget:.2f} M")
#     col2.metric("📈 Ventes prédites", f"{prediction:.2f} M")
#     col3.metric("🚀 ROI estimé", f"{roi:.2f}")

#     st.divider()

#     st.subheader("📌 Scénario testé")
#     st.dataframe(input_data, use_container_width=True)

#     st.subheader("📊 Aperçu des données marketing")
#     st.dataframe(df.head(20), use_container_width=True)

#     st.subheader("📈 Budget TV vs ventes")
#     st.scatter_chart(df, x="TV", y="Sales")

#     st.subheader("📈 Budget Radio vs ventes")
#     st.scatter_chart(df, x="Radio", y="Sales")

#     st.subheader("📈 Budget Social Media vs ventes")
#     st.scatter_chart(df, x="Social Media", y="Sales")

#     comparison_df = safe_load_csv(COMPARISON_PATH)

#     if comparison_df is not None:
#         st.subheader("🏆 Comparaison des modèles")
#         st.dataframe(comparison_df, use_container_width=True)
#         st.bar_chart(comparison_df.set_index("model")[["RMSE", "MAE"]])
#     else:
#         st.warning("Le fichier de comparaison des modèles n’est pas encore disponible.")

#     importance_df = safe_load_csv(IMPORTANCE_PATH)

#     if importance_df is not None:
#         st.subheader("🧠 Importance des variables")
#         st.dataframe(importance_df, use_container_width=True)
#         st.bar_chart(importance_df.set_index("feature")["importance_mean"])
#     else:
#         st.warning("Le fichier d’importance des variables n’est pas encore disponible.")

#     st.divider()

#     st.info(
#         "Interprétation : plus le ROI est élevé, plus la combinaison budgétaire "
#         "semble efficace pour générer des ventes. Cette prédiction reste une aide "
#         "à la décision, pas une preuve de causalité."
#     )


# if __name__ == "__main__":
#     main()

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.logger import get_logger

logger = get_logger("streamlit_dashboard")

DATA_PATH = Path("data/marketing_and_sales.csv")
MODEL_PATH = Path("models/best_model.pkl")
COMPARISON_PATH = Path("reports/model_comparison.csv")
IMPORTANCE_PATH = Path("reports/feature_importance.csv")

st.set_page_config(
    page_title="Marketing ROI Optimizer",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["TV", "Radio", "Social Media", "Influencer", "Sales"])
    return df


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data
def safe_load_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def predict_sales(model, tv: float, radio: float, social_media: float, influencer: str) -> float:
    input_data = pd.DataFrame([
        {
            "TV": tv,
            "Radio": radio,
            "Social Media": social_media,
            "Influencer": influencer,
        }
    ])
    return float(model.predict(input_data)[0])


def compute_roi(sales: float, tv: float, radio: float, social_media: float) -> float:
    total_budget = tv + radio + social_media
    return sales / total_budget if total_budget > 0 else 0.0


def build_scenario_table(model, tv, radio, social_media, influencer):
    base_sales = predict_sales(model, tv, radio, social_media, influencer)
    base_budget = tv + radio + social_media
    base_roi = compute_roi(base_sales, tv, radio, social_media)

    scenarios = []
    changes = {
        "TV +10%": {"TV": tv * 1.10, "Radio": radio, "Social Media": social_media},
        "Radio +10%": {"TV": tv, "Radio": radio * 1.10, "Social Media": social_media},
        "Social Media +10%": {"TV": tv, "Radio": radio, "Social Media": social_media * 1.10},
        "Tous budgets +10%": {"TV": tv * 1.10, "Radio": radio * 1.10, "Social Media": social_media * 1.10},
    }

    for name, vals in changes.items():
        sales = predict_sales(model, vals["TV"], vals["Radio"], vals["Social Media"], influencer)
        budget = vals["TV"] + vals["Radio"] + vals["Social Media"]
        roi = compute_roi(sales, vals["TV"], vals["Radio"], vals["Social Media"])
        scenarios.append(
            {
                "Scénario": name,
                "Budget total": budget,
                "Ventes prédites": sales,
                "Δ ventes": sales - base_sales,
                "ROI": roi,
                "Δ ROI": roi - base_roi,
            }
        )

    return pd.DataFrame(scenarios)


def build_response_curve(model, df, channel: str, tv, radio, social_media, influencer):
    min_val = float(df[channel].min())
    max_val = float(df[channel].max())
    values = pd.Series([min_val + i * (max_val - min_val) / 30 for i in range(31)])

    rows = []
    for value in values:
        current_tv = value if channel == "TV" else tv
        current_radio = value if channel == "Radio" else radio
        current_social = value if channel == "Social Media" else social_media
        sales = predict_sales(model, current_tv, current_radio, current_social, influencer)
        roi = compute_roi(sales, current_tv, current_radio, current_social)
        rows.append({channel: value, "Ventes prédites": sales, "ROI": roi})

    return pd.DataFrame(rows)


def main():
    st.title("📊 Marketing ROI Optimizer")
    st.caption("Transformer les budgets marketing en décisions mesurables : ventes prévues, ROI et leviers d'action.")

    try:
        df = load_data()
        model = load_model()
    except Exception as error:
        logger.exception("Erreur critique au démarrage du dashboard")
        st.error(f"Erreur au chargement : {error}")
        st.stop()

    comparison_df = safe_load_csv(COMPARISON_PATH)
    importance_df = safe_load_csv(IMPORTANCE_PATH)

    st.sidebar.header("🎛️ Simuler un scénario")
    st.sidebar.info(
        "Modifiez les budgets pour observer l'impact estimé sur les ventes et le ROI. "
        "Le ROI peut baisser si le budget augmente plus vite que les ventes."
    )

    tv = st.sidebar.slider("Budget TV", float(df["TV"].min()), float(df["TV"].max()), float(df["TV"].mean()))
    radio = st.sidebar.slider("Budget Radio", float(df["Radio"].min()), float(df["Radio"].max()), float(df["Radio"].mean()))
    social_media = st.sidebar.slider(
        "Budget Social Media",
        float(df["Social Media"].min()),
        float(df["Social Media"].max()),
        float(df["Social Media"].mean()),
    )
    influencer = st.sidebar.selectbox("Type d'influenceur", sorted(df["Influencer"].unique()))

    prediction = predict_sales(model, tv, radio, social_media, influencer)
    total_budget = tv + radio + social_media
    roi = compute_roi(prediction, tv, radio, social_media)

    logger.info(
        "Prédiction Streamlit | "
        f"TV={tv:.2f}, Radio={radio:.2f}, Social Media={social_media:.2f}, "
        f"Influencer={influencer}, Sales={prediction:.2f}, ROI={roi:.3f}"
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Budget total", f"{total_budget:.2f} M")
    kpi2.metric("Ventes prédites", f"{prediction:.2f} M")
    kpi3.metric("ROI estimé", f"{roi:.2f}")
    kpi4.metric("Modèle utilisé", "Best model")

    st.markdown(
        """
        **Lecture business :** l'application aide un responsable marketing à tester un mix budgétaire avant lancement.  
        Elle ne donne pas une vérité causale absolue : elle reproduit les relations apprises dans les données historiques.
        """
    )

    tab_decision, tab_models, tab_data, tab_limits = st.tabs(
        ["💼 Décision métier", "🤖 Modèles", "📊 Données", "⚠️ Limites & interprétation"]
    )

    with tab_decision:
        st.subheader("Impact incrémental des budgets")
        scenario_df = build_scenario_table(model, tv, radio, social_media, influencer)
        st.dataframe(
            scenario_df.style.format(
                {
                    "Budget total": "{:.2f}",
                    "Ventes prédites": "{:.2f}",
                    "Δ ventes": "{:+.2f}",
                    "ROI": "{:.2f}",
                    "Δ ROI": "{:+.2f}",
                }
            ),
            use_container_width=True,
        )

        best_sales_row = scenario_df.sort_values("Δ ventes", ascending=False).iloc[0]
        best_roi_row = scenario_df.sort_values("Δ ROI", ascending=False).iloc[0]
        c1, c2 = st.columns(2)
        c1.success(f"Meilleur gain de ventes simulé : {best_sales_row['Scénario']} ({best_sales_row['Δ ventes']:+.2f} M).")
        c2.info(f"Meilleure évolution du ROI simulée : {best_roi_row['Scénario']} ({best_roi_row['Δ ROI']:+.2f}).")

        if (scenario_df["Δ ventes"] < 0).any():
            st.warning(
                "Certaines hausses de budget réduisent les ventes prédites. Ce n'est pas forcément un bug : "
                "dans ce dataset, TV explique presque toute la vente, tandis que Radio et Social Media sont fortement corrélés à TV. "
                "À budget TV constant, leur effet incrémental estimé peut donc être faible ou négatif."
            )

        st.subheader("Courbes de réponse : que se passe-t-il si un seul canal varie ?")
        channel = st.selectbox("Canal à analyser", ["TV", "Radio", "Social Media"])
        curve_df = build_response_curve(model, df, channel, tv, radio, social_media, influencer)
        col_a, col_b = st.columns(2)
        col_a.line_chart(curve_df, x=channel, y="Ventes prédites")
        col_b.line_chart(curve_df, x=channel, y="ROI")

        st.caption(
            "Ces courbes tiennent les autres budgets constants. Elles servent à interpréter le comportement du modèle, "
            "pas à prouver une causalité marketing."
        )

    with tab_models:
        st.subheader("Comparaison quantitative des modèles")
        if comparison_df is not None:
            st.dataframe(comparison_df.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.4f}"}), use_container_width=True)
            st.bar_chart(comparison_df.set_index("model")[["RMSE", "MAE"]])
            best = comparison_df.sort_values("RMSE").iloc[0]
            st.success(
                f"Modèle retenu automatiquement : {best['model']} — RMSE={best['RMSE']:.3f}, R²={best['R2']:.4f}."
            )
            st.markdown(
                """
                **Pourquoi plusieurs modèles ?**  
                - Linear Regression : baseline simple et interprétable.  
                - Random Forest : capture des relations non linéaires.  
                - Gradient Boosting : compromis performance/complexité.  
                - MLP Regressor : modèle Deep Learning obligatoire pour comparer avec le ML classique.
                """
            )
        else:
            st.warning("Fichier reports/model_comparison.csv absent. Lancez : python -m src.train_models")

        st.subheader("Importance des variables")
        if importance_df is not None:
            st.dataframe(importance_df.style.format({"importance_mean": "{:.4f}", "importance_std": "{:.4f}"}), use_container_width=True)
            st.bar_chart(importance_df.set_index("feature")["importance_mean"])
            top_feature = importance_df.sort_values("importance_mean", ascending=False).iloc[0]["feature"]
            st.info(
                f"Insight principal : {top_feature} est le levier le plus déterminant dans les prédictions du modèle."
            )
        else:
            st.warning("Fichier reports/feature_importance.csv absent. Lancez : python -m src.evaluate")

    with tab_data:
        st.subheader("Vue dataset")
        st.dataframe(df.head(30), use_container_width=True)

        st.subheader("Corrélations observées")
        corr = df[["TV", "Radio", "Social Media", "Sales"]].corr()
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)
        st.markdown(
            """
            **Point clé :** dans ce dataset, les ventes sont presque parfaitement corrélées au budget TV.  
            Radio et Social Media sont aussi corrélés à TV : le modèle peut donc considérer qu'ils apportent peu d'information additionnelle quand TV est déjà connue.
            """
        )

        st.subheader("Relations historiques budgets → ventes")
        b1, b2, b3 = st.columns(3)
        b1.scatter_chart(df, x="TV", y="Sales")
        b2.scatter_chart(df, x="Radio", y="Sales")
        b3.scatter_chart(df, x="Social Media", y="Sales")

    with tab_limits:
        st.subheader("Pourquoi le ROI peut diminuer quand le budget augmente ?")
        st.markdown(
            """
            Le ROI est calculé comme : **ventes prédites / budget total**.  
            Même si les ventes augmentent légèrement, le ROI peut baisser si le budget augmente plus vite que les ventes.

            **Pourquoi les ventes peuvent parfois baisser quand Radio ou Social Media augmente ?**  
            Dans le modèle actuel, les coefficients/impacts de Radio et Social Media sont très faibles car la variable TV domine la prédiction.  
            Cela vient probablement de la structure du dataset : TV explique presque toute la variation de Sales.
            """
        )
        st.warning(
            "Conclusion business : l'application doit être utilisée comme outil de simulation et d'aide à la décision. "
            "Elle met en évidence qu'il faudrait collecter plus de données réelles pour mieux mesurer l'effet propre de chaque canal."
        )
        st.markdown(
            """
            **Améliorations possibles :**  
            - collecter plus de campagnes réelles ;  
            - ajouter des variables temporelles, saisonnalité, secteur, objectif campagne ;  
            - utiliser SHAP pour expliquer localement chaque prédiction ;  
            - utiliser des modèles avec contraintes métier si l'on veut imposer qu'un budget plus élevé ne baisse jamais les ventes.
            """
        )


if __name__ == "__main__":
    main()
