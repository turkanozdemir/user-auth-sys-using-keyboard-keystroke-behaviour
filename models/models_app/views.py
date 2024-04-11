import base64
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             matthews_corrcoef, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC


def generate_classification_report(y_test, y_predict):
    return {
        'Accuracy': accuracy_score(y_test, y_predict),
        'Precision': precision_score(y_test, y_predict, average='weighted'),
        'Recall': recall_score(y_test, y_predict, average='weighted'),
        'F1-Score': f1_score(y_test, y_predict, average='weighted'),
        'Matthews Correlation Coefficient': matthews_corrcoef(y_test, y_predict),
    }


def plot_confusion_matrix(model_name, y_test, y_predict):
    unique_classes = sorted(set(y_test))
    class_labels = [f"User {class_num + 1}" for class_num in unique_classes]

    plt.figure(figsize=(10, 7))
    sns.set_context('paper', font_scale=1.0)
    sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, cmap='Blues', xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.yticks(rotation=0)
    plt.title(model_name + ' Confusion Matrix', fontsize=15)

    conf_buffer = BytesIO()
    plt.savefig(conf_buffer, format='png')
    conf_buffer.seek(0)
    conf_image = base64.b64encode(conf_buffer.read()).decode('utf-8')
    # img_tag = f'<img src="data:image/png;base64,{conf_image}">'
    plt.close()

    return conf_image


def generate_correlation_matrix(data):
    correlation_matrix_data = data.drop('label', axis=1)
    correlation_matrix = correlation_matrix_data.corr()

    plt.figure(figsize=(10, 7))
    sns.set_context('paper', font_scale=0.8)
    sns.heatmap(correlation_matrix, annot=True, cmap='Blues')
    plt.title('Correlation Matrix')

    corr_buffer = BytesIO()
    plt.savefig(corr_buffer, format='png')
    corr_buffer.seek(0)
    corr_image = base64.b64encode(corr_buffer.read()).decode('utf-8')
    plt.close()

    return corr_image


def generate_feature_importance(data):
    features = ['H', 'DD', 'UD', 'key_stroke_average', 'back_space_count', 'used_caps', 'shift_left_favored']

    x = data.drop('label', axis=1).values
    y = data['label'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier()
    model.fit(x, y)

    plt.figure(figsize=(10, 7))
    plt.barh(features[::-1], model.feature_importances_[::-1])
    plt.title('Feature Importance (Decision Tree)')
    plt.xlabel('Importance')
    plt.ylabel('Features')

    imp_buffer = BytesIO()
    plt.savefig(imp_buffer, format='png')
    imp_buffer.seek(0)

    importance_image = base64.b64encode(imp_buffer.read()).decode('utf-8')

    plt.close()

    return importance_image


def plot_roc_curve(name, model_probs, y_test_bin, y):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(np.unique(y))):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], model_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), model_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    plt.plot(fpr_micro, tpr_micro, color='navy', lw=2, label=name + ' (area = {:.2f})'.format(roc_auc_micro))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro-average ROC Curve', fontsize=15)
    plt.legend(loc='lower right')

    roc_buffer = BytesIO()
    plt.savefig(roc_buffer, format='png')
    roc_buffer.seek(0)
    roc_curve_image = base64.b64encode(roc_buffer.read()).decode('utf-8')
    plt.close()

    return roc_curve_image


@csrf_exempt
def svm_predictions(request):
    if request.method == 'POST' and request.FILES['file']:
        split_ratio = float(request.POST.get('splitRatio'))
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)

        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

        x = data.drop('label', axis=1).values
        y = data['label'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        svm_model.fit(x_train, y_train)
        svm_predict = svm_model.predict(x_test)

        results = {
            'analyzeReport': generate_classification_report(y_test, svm_predict),
            'correlationMatrixImage': generate_correlation_matrix(data),
            'featureImportanceImage': generate_feature_importance(data),
            'confusionMatrix': plot_confusion_matrix('SVM', y_test, svm_predict),
            'rocCurve': plot_roc_curve("SVM", svm_model.predict_proba(x_test),
                                       label_binarize(y_test, classes=np.unique(y)), y)
        }

        return JsonResponse(results)
    else:
        return JsonResponse({'error': 'No file found or method is not POST'}, status=400)


@csrf_exempt
def mlp_predictions(request):
    if request.method == 'POST' and request.FILES['file']:
        split_ratio = float(request.POST.get('splitRatio'))
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)

        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

        x = data.drop('label', axis=1).values
        y = data['label'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        mlp_model = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=10000, random_state=1)
        mlp_model.fit(x_train, y_train)
        mlp_predict = mlp_model.predict(x_test)

        results = {
            'analyzeReport': generate_classification_report(y_test, mlp_predict),
            'correlationMatrixImage': generate_correlation_matrix(data),
            'featureImportanceImage': generate_feature_importance(data),
            'confusionMatrix': plot_confusion_matrix('MLP', y_test, mlp_predict),
            'rocCurve': plot_roc_curve("MLP", mlp_model.predict_proba(x_test),
                                       label_binarize(y_test, classes=np.unique(y)), y)
        }

        return JsonResponse(results)
    else:
        return JsonResponse({'error': 'No file found or method is not POST'}, status=400)


@csrf_exempt
def knn_predictions(request):
    if request.method == 'POST' and request.FILES['file']:
        split_ratio = float(request.POST.get('splitRatio'))
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)

        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

        x = data.drop('label', axis=1).values
        y = data['label'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        #  Use cross validation and GridSearchCV to select the optimal k value
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
        grid_search.fit(x_train_scaled, y_train)
        best_k = grid_search.best_params_['n_neighbors']

        knn_model = OneVsRestClassifier(KNeighborsClassifier())
        knn_model.fit(x_train_scaled, y_train)
        knn_predict = knn_model.predict(x_test_scaled)

        results = {
            'analyzeReport': generate_classification_report(y_test, knn_predict),
            'correlationMatrixImage': generate_correlation_matrix(data),
            'featureImportanceImage': generate_feature_importance(data),
            'confusionMatrix': plot_confusion_matrix('KNN', y_test, knn_predict),
            'rocCurve': plot_roc_curve("KNN", knn_model.predict_proba(x_test),
                                       label_binarize(y_test, classes=np.unique(y)), y)
        }

        return JsonResponse(results)
    else:
        return JsonResponse({'error': 'No file found or method is not POST'}, status=400)


@csrf_exempt
def lr_predictions(request):
    if request.method == 'POST' and request.FILES['file']:
        split_ratio = float(request.POST.get('splitRatio'))
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)

        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

        x = data.drop('label', axis=1).values
        y = data['label'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=10)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # Find the best hyperparameters using GridSearchCV
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        lr_model = LogisticRegression(class_weight='balanced', multi_class='auto', solver='lbfgs', C=1.0)

        grid_search = GridSearchCV(lr_model, param_grid, cv=5)
        grid_search.fit(x_train, y_train)
        # Choose the best model
        lr_model = grid_search.best_estimator_
        lr_predict = lr_model.predict(x_test)

        results = {
            'analyzeReport': generate_classification_report(y_test, lr_predict),
            'correlationMatrixImage': generate_correlation_matrix(data),
            'featureImportanceImage': generate_feature_importance(data),
            'confusionMatrix': plot_confusion_matrix('LR', y_test, lr_predict),
            'rocCurve': plot_roc_curve("LR", lr_model.predict_proba(x_test),
                                       label_binarize(y_test, classes=np.unique(y)), y)
        }

        return JsonResponse(results)
    else:
        return JsonResponse({'error': 'No file found or method is not POST'}, status=400)


@csrf_exempt
def all_models(request):
    if request.method == 'POST' and request.FILES['file']:
        split_ratio = float(request.POST.get('splitRatio'))
        uploaded_file = request.FILES['file']
        data = pd.read_csv(uploaded_file)

        label_encoder = LabelEncoder()
        data['label'] = label_encoder.fit_transform(data['label'])

        x = data.drop('label', axis=1).values
        y = data['label'].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_ratio, random_state=42)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # SVM Model
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
        svm_model.fit(x_train, y_train)
        svm_predict = svm_model.predict(x_test)
        svm_accuracy = accuracy_score(y_test, svm_predict)

        # MLP Model
        mlp_model = MLPClassifier(hidden_layer_sizes=(200, 100, 50), max_iter=10000, random_state=1)
        mlp_model.fit(x_train, y_train)
        mlp_predict = mlp_model.predict(x_test)
        mlp_accuracy = accuracy_score(y_test, mlp_predict)

        # KNN Model
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')

        knn_model.fit(x_train, y_train)
        knn_predict = knn_model.predict(x_test)
        knn_accuracy = accuracy_score(y_test, knn_predict)

        # Logistic Regression Model
        lr_model = LogisticRegression(class_weight='balanced', multi_class='auto', solver='lbfgs', C=1.0)
        lr_model.fit(x_train, y_train)
        lr_predict = lr_model.predict(x_test)
        lr_accuracy = accuracy_score(y_test, lr_predict)

        Accuracy = [
            mlp_accuracy,
            knn_accuracy,
            svm_accuracy,
            lr_accuracy
        ]

        model = ["MLP", "KNN", "SVM", "LR"]
        models_acc = []

        for i in range(0, 4):
            models_acc.append([model[i], Accuracy[i]])

        datasetAcc = pd.DataFrame(models_acc)
        datasetAcc.columns = ['Model', 'Accuracy']
        datasetAcc.sort_values(by='Accuracy', ascending=False, inplace=True)
        datasetAcc.reset_index(drop=True, inplace=True)

        plt.figure(figsize=(15, 7))
        sns.set(font_scale=1.5)
        sns.barplot(x='Model', y='Accuracy', data=datasetAcc)
        plt.title('Accuracy of Models', fontsize=15)
        plt.ylim(0, 1)

        imp_buffer = BytesIO()
        plt.savefig(imp_buffer, format='png')
        imp_buffer.seek(0)
        comparison = base64.b64encode(imp_buffer.read()).decode('utf-8')
        plt.close()

        # -------------------------------------------------------------------------------------------------------------------------------

        # LR
        lr_probs = lr_model.predict_proba(x_test)
        y_test_bin_lr = label_binarize(y_test, classes=list(range(len(set(y)))))

        fpr_lr = dict()
        tpr_lr = dict()
        roc_auc_lr = dict()

        for i in range(len(set(y))):
            fpr_lr[i], tpr_lr[i], _ = roc_curve(y_test_bin_lr[:, i], lr_probs[:, i])
            roc_auc_lr[i] = auc(fpr_lr[i], tpr_lr[i])

        fpr_micro_lr, tpr_micro_lr, _ = roc_curve(y_test_bin_lr.ravel(), lr_probs.ravel())
        roc_auc_micro_lr = auc(fpr_micro_lr, tpr_micro_lr)

        # SVM
        svm_probs = svm_model.decision_function(x_test)
        y_test_bin_svm = label_binarize(y_test, classes=np.unique(y))

        fpr_svm = dict()
        tpr_svm = dict()
        roc_auc_svm = dict()

        for i in range(len(np.unique(y))):
            fpr_svm[i], tpr_svm[i], _ = roc_curve(y_test_bin_svm[:, i], svm_probs[:, i])
            roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])

        fpr_micro_svm, tpr_micro_svm, _ = roc_curve(y_test_bin_svm.ravel(), svm_probs.ravel())
        roc_auc_micro_svm = auc(fpr_micro_svm, tpr_micro_svm)

        # KNN
        knn_probs = knn_model.fit(x_train, y_train).predict_proba(x_test)
        y_test_bin_knn = label_binarize(y_test, classes=np.unique(y))

        fpr_knn = dict()
        tpr_knn = dict()
        roc_auc_knn = dict()

        for i in range(len(np.unique(y))):
            fpr_knn[i], tpr_knn[i], _ = roc_curve(y_test_bin_knn[:, i], knn_probs[:, i])
            roc_auc_knn[i] = auc(fpr_knn[i], tpr_knn[i])

        fpr_micro_knn, tpr_micro_knn, _ = roc_curve(y_test_bin_knn.ravel(), knn_probs.ravel())
        roc_auc_micro_knn = auc(fpr_micro_knn, tpr_micro_knn)

        # MLP
        mlp_probs = mlp_model.predict_proba(x_test)
        y_test_bin_mlp = label_binarize(y_test, classes=np.unique(y))

        fpr_mlp = dict()
        tpr_mlp = dict()
        roc_auc_mlp = dict()

        for i in range(len(np.unique(y))):
            fpr_mlp[i], tpr_mlp[i], _ = roc_curve(y_test_bin_mlp[:, i], mlp_probs[:, i])
            roc_auc_mlp[i] = auc(fpr_mlp[i], tpr_mlp[i])

        fpr_micro_mlp, tpr_micro_mlp, _ = roc_curve(y_test_bin_mlp.ravel(), mlp_probs.ravel())
        roc_auc_micro_mlp = auc(fpr_micro_mlp, tpr_micro_mlp)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr_micro_mlp, tpr_micro_mlp, color='red', lw=2,
                 label='MLP (area = {:.2f})'.format(roc_auc_micro_mlp))  # For MLP
        plt.plot(fpr_micro_knn, tpr_micro_knn, color='orange', lw=2,
                 label='KNN (area = {:.2f})'.format(roc_auc_micro_knn))  # For KNN
        plt.plot(fpr_micro_svm, tpr_micro_svm, color='green', lw=2,
                 label='SVM (area = {:.2f})'.format(roc_auc_micro_svm))  # For SVM
        plt.plot(fpr_micro_lr, tpr_micro_lr, color='blue', lw=2,
                 label='LR (area = {:.2f})'.format(roc_auc_micro_lr))  # For LR

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Micro-average ROC Curve for All Models', fontsize=15)
        plt.legend(loc='lower right')

        roc_buffer = BytesIO()
        plt.savefig(roc_buffer, format='png')
        roc_buffer.seek(0)
        rocCurve = base64.b64encode(roc_buffer.read()).decode('utf-8')
        plt.close()

        results = {
            'comparison': comparison,
            'rocCurve': rocCurve,
            'correlationMatrixImage': generate_correlation_matrix(data),
            'featureImportanceImage': generate_feature_importance(data),
        }

        return JsonResponse(results)
