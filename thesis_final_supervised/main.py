from pathlib import Path

import pandas as pd
import torch

from thesis_final_supervised.src.dataset.Dataset_Constructor import create_dataset
from thesis_final_supervised.src.utils.evaluation_data import (
    gather_split_writer_signature_paths,
    sample_reference_and_query_genuine_paths,
)
from thesis_final_supervised.src.utils.evaluation_embeddings import (
    build_writer_prototypes,
    encode_reference_genuine_paths,
)
from thesis_final_supervised.src.utils.evaluation_metrics import (
    collect_prototype_scores,
    compute_fixed_threshold_metrics,
    select_validation_threshold,
)
from thesis_final_supervised.src.utils.evaluation_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
)
from thesis_final_supervised.src.utils.model_loader import load_verification_model
from thesis_final_supervised.src.utils.prototype_pca_visualization import (
    plot_writer_prototypes_pca,
)


def build_split_evaluation_artifacts(
        model,
        split_summary_path: str,
        dataset_root: str,
        split_key: str,
        device: torch.device,
        num_reference_genuine: int = 5,
        num_forgery_queries: int = None,
        seed: int = 2026,
        target_size=(512, 512),
):
    writer_to_signatures = gather_split_writer_signature_paths(
        split_summary_path=split_summary_path,
        dataset_root=dataset_root,
        split_key=split_key,
    )
    writer_to_protocol = sample_reference_and_query_genuine_paths(
        writer_to_signatures=writer_to_signatures,
        num_reference_genuine=num_reference_genuine,
        num_forgery_queries=num_forgery_queries,
        seed=seed,
    )
    writer_to_reference_embeddings = encode_reference_genuine_paths(
        model=model,
        writer_to_protocol=writer_to_protocol,
        device=device,
        target_size=target_size,
    )
    writer_to_prototype = build_writer_prototypes(writer_to_reference_embeddings)
    score_df, protocol_df = collect_prototype_scores(
        model=model,
        writer_to_protocol=writer_to_protocol,
        writer_to_prototype=writer_to_prototype,
        device=device,
        target_size=target_size,
    )
    return writer_to_protocol, writer_to_prototype, score_df, protocol_df


def main():

    # code for creating the dataset for the downstream verification training
    writer_split_json_path = r'D:\Learning_Education_and_Application\Thesis\thesis_final_supervised\misc\BHSig260_Hindi_split_summary.json'
    dataset_root = r'D:\Learning_Education_and_Application\Datasets\Signature Dataset\BHSig260-Hindi\BHSig260-Hindi'

    create_dataset(
        split_summary_path=writer_split_json_path,
        dataset_root=dataset_root
    )


    # figures_directory = Path("figures")
    # csv_directory = Path("csv_files")
    # validation_roc_figure_path = figures_directory / "validation_averaged_prototype_roc_curve_bengali_dataset.png"
    # test_pca_figure_path = figures_directory / "test_writer_prototypes_pca.png"
    # confusion_matrix_figure_path = figures_directory / "test_confusion_matrix_averaged_prototype_bengali_dataset.png"
    # test_metrics_csv_path = csv_directory / "test_metrics_summary_averaged_prototype_bengali_dataset.csv"
    # split_summary_path = r"D:\Learning_Education_and_Application\Thesis\thesis_final_supervised\misc\BHSig260_Bengali_split_summary.json"
    # dataset_root = r"D:\Learning_Education_and_Application\Datasets\Signature Dataset\BHSig260-Bengali\BHSig260-Bengali"
    # verification_model_path = r"D:\Learning_Education_and_Application\Thesis_results\Downstream_Verification_Result_BHSig260_Bengali\saved_models\best_triplet_loss_verification_json_writer_disjoint.pt"
    # target_size = (512, 512)
    # num_reference_genuine = 5
    # seed = 2026
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # verification_model = load_verification_model(
    #     device=device,
    #     model_path=verification_model_path,
    # ).to(device)
    # verification_model.eval()
    #
    # val_protocol, val_prototypes, val_score_df, val_protocol_df = build_split_evaluation_artifacts(
    #     model=verification_model,
    #     split_summary_path=split_summary_path,
    #     dataset_root=dataset_root,
    #     split_key='val_writers',
    #     device=device,
    #     num_reference_genuine=num_reference_genuine,
    #     num_forgery_queries=None,
    #     seed=seed,
    #     target_size=target_size,
    # )
    # val_roc_df, val_roc_auc, threshold_info = select_validation_threshold(val_score_df)
    # plot_roc_curve(
    #     roc_df=val_roc_df,
    #     roc_auc=val_roc_auc,
    #     threshold_info=threshold_info,
    #     title="Validation ROC Curve",
    #     save_path=validation_roc_figure_path,
    # )
    #
    # test_protocol, test_prototypes, test_score_df, test_protocol_df = build_split_evaluation_artifacts(
    #     model=verification_model,
    #     split_summary_path=split_summary_path,
    #     dataset_root=dataset_root,
    #     split_key='test_writers',
    #     device=device,
    #     num_reference_genuine=num_reference_genuine,
    #     num_forgery_queries=19,
    #     seed=seed,
    #     target_size=target_size,
    # )
    # test_metrics = compute_fixed_threshold_metrics(
    #     score_df=test_score_df,
    #     distance_threshold=threshold_info['best_distance_threshold'],
    # )
    #
    # plot_writer_prototypes_pca(
    #     writer_to_prototype=test_prototypes,
    #     title="PCA of Test Writer Prototypes",
    #     annotate_points=True,
    #     save_path=test_pca_figure_path,
    # )
    # plot_confusion_matrix(
    #     score_df=test_score_df,
    #     distance_threshold=threshold_info['best_distance_threshold'],
    #     title="Confusion matrix",
    #     save_path=confusion_matrix_figure_path,
    # )
    #
    # csv_directory.mkdir(parents=True, exist_ok=True)
    # test_metrics_report_df = pd.DataFrame([{
    #     'TPR_Recall_Sensitivity': test_metrics['tpr_recall_sensitivity'],
    #     'FAR_FPR': test_metrics['far'],
    #     'FRR_FNR': test_metrics['frr'],
    #     'Accuracy': test_metrics['accuracy'],
    #     'Precision': test_metrics['precision'],
    #     'Specificity_TNR': test_metrics['specificity_tnr'],
    #     'F1_Score': test_metrics['f1_score'],
    #     'Distance_Threshold': test_metrics['distance_threshold'],
    #     'True_Negative': test_metrics['true_negative'],
    #     'False_Positive': test_metrics['false_positive'],
    #     'False_Negative': test_metrics['false_negative'],
    #     'True_Positive': test_metrics['true_positive'],
    #     'Num_Samples': test_metrics['num_samples'],
    # }])
    # test_metrics_report_df.to_csv(test_metrics_csv_path, index=False)
    #
    # print("Validation evaluation")
    # print(f"Validation writers: {len(val_protocol)}")
    # print(f"Validation protocol rows: {len(val_protocol_df)}")
    # print(f"Validation score rows: {len(val_score_df)}")
    # print(f"Validation ROC AUC: {val_roc_auc:.6f}")
    # print(f"Validation best distance threshold: {threshold_info['best_distance_threshold']:.6f}")
    # print(f"Validation EER estimate: {threshold_info['eer_estimate']:.6f}")
    # print()
    # print("Test evaluation")
    # print(f"Test writers: {len(test_protocol)}")
    # print(f"Test protocol rows: {len(test_protocol_df)}")
    # print(f"Test score rows: {len(test_score_df)}")
    # print(f"Applied distance threshold: {test_metrics['distance_threshold']:.6f}")
    # print(f"Test accuracy: {test_metrics['accuracy']:.6f}")
    # print(f"Test FAR: {test_metrics['far']:.6f}")
    # print(f"Test FRR: {test_metrics['frr']:.6f}")
    # print(f"Test specificity/TNR: {test_metrics['specificity_tnr']:.6f}")
    # print(f"Test F1 score: {test_metrics['f1_score']:.6f}")
    # print(f"Test precision: {test_metrics['precision']:.6f}")
    # print(f"Test recall: {test_metrics['recall']:.6f}")
    # print(
    #     "Confusion counts: "
    #     f"TN={test_metrics['true_negative']} "
    #     f"FP={test_metrics['false_positive']} "
    #     f"FN={test_metrics['false_negative']} "
    #     f"TP={test_metrics['true_positive']}"
    # )


if __name__ == "__main__":
    main()
