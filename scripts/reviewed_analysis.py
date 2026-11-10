"""A reviewed educational analysis; original notebook/runner remain historical.

Use --split-strategy profile for a separate exact-profile generalization check.
Profiles are NOT patient identifiers. This previously explored data is not a
fresh external test of clinical performance.
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold,
    StratifiedKFold, cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [f"attr{i}" for i in range(1, 7)]
LABELS = [-1, 1]


def load_data(path):
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ["class"]:
        raise ValueError("Expected exactly attr1..attr6 followed by class.")
    x = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    raw = pd.to_numeric(frame["class"], errors="raise")
    if raw.isna().any() or not raw.isin(LABELS).all():
        raise ValueError("class must contain only recorded -1 and +1 labels.")
    if np.isinf(x.to_numpy()).any() or x.isna().all().any():
        raise ValueError("Features contain infinity or an entirely missing column.")
    y = raw.astype(int)
    if set(y.unique()) != set(LABELS):
        raise ValueError("Both classes are required.")
    keys = [tuple(None if pd.isna(v) else float(v) for v in row) for row in x.to_numpy()]
    lookup = {}
    groups = np.array([lookup.setdefault(k, len(lookup)) for k in keys])
    return x, y, groups


def group_splits(x, y, groups, n_splits, seed):
    # Deterministic size-aware allocation avoids an extremely small shuffled fold
    # in this dataset. The seed controls models, not this optional group split.
    splits = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=False).split(x, y, groups))
    for train, valid in splits:
        assert not (set(groups[train]) & set(groups[valid]))
        if set(y.iloc[train].unique()) != set(LABELS) or set(y.iloc[valid].unique()) != set(LABELS):
            raise ValueError("A group fold lacks one class; use fewer folds or more data.")
    return splits


def evaluate(y_true, predictions, positive_scores=None):
    if not isinstance(predictions, pd.Series) or not y_true.index.equals(predictions.index):
        raise ValueError("Prediction row IDs must exactly match target row IDs.")
    if not predictions.isin(LABELS).all() or set(y_true.unique()) != set(LABELS):
        raise ValueError("Expected -1/+1 predictions and both target classes.")
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=LABELS).ravel()
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    result = {
        "n":len(y_true), "positive_support":int(tp+fn),
        "confusion_matrix_labels":LABELS, "confusion_matrix":[[int(tn),int(fp)],[int(fn),int(tp)]],
        "accuracy":float(accuracy_score(y_true,predictions)),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,predictions)),
        "positive_precision":float(precision_score(y_true,predictions,pos_label=1,zero_division=0)),
        "positive_recall":float(recall), "specificity":float(specificity),
        "positive_f1":float(f1_score(y_true,predictions,pos_label=1,zero_division=0)),
        "macro_f1":float(f1_score(y_true,predictions,average="macro",zero_division=0)),
        "g_mean":float(np.sqrt(recall*specificity)),
    }
    if positive_scores is not None:
        result["roc_auc"]=float(roc_auc_score(y_true==1,positive_scores))
        result["average_precision"]=float(average_precision_score(y_true==1,positive_scores))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=Path(__file__).resolve().parents[1]/"notebooks/Mammography.csv")
    parser.add_argument("--output-json",type=Path,default=Path("reports/results/reviewed_metrics.json"))
    parser.add_argument("--split-strategy",choices=["row","profile"],default="row")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--cv-splits",type=int,default=5)
    parser.add_argument("--jobs",type=int,default=1)
    args=parser.parse_args()
    x,y,groups=load_data(args.data)
    if args.split_strategy=="profile":
        train,test=group_splits(x,y,groups,5,args.seed)[0]
        cv=group_splits(x.iloc[train],y.iloc[train],groups[train],args.cv_splits,100)
    else:
        train,test=train_test_split(np.arange(len(x)),test_size=.2,stratify=y,random_state=args.seed)
        cv=list(StratifiedKFold(n_splits=args.cv_splits,shuffle=True,random_state=100).split(x.iloc[train],y.iloc[train]))
    x_train,x_test,y_train,y_test=x.iloc[train],x.iloc[test],y.iloc[train],y.iloc[test]
    preprocessing=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    def rf(weight=None):
        return RandomForestClassifier(n_estimators=100,random_state=args.seed,class_weight=weight,n_jobs=args.jobs)
    candidates={
        "baseline_rf":Pipeline([("preprocessor",clone(preprocessing)),("classifier",rf())]),
        "weighted_rf":Pipeline([("preprocessor",clone(preprocessing)),("classifier",rf("balanced"))]),
        # All six inputs are continuous. Fit preprocessing and resampling inside CV.
        # Default SMOTE already balances; default undersampling afterward is a no-op.
        "smote_rf":ImbPipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("oversample",SMOTE(random_state=args.seed)),("classifier",rf())]),
    }
    scores={}
    for name,model in candidates.items():
        values=cross_val_score(model,x_train,y_train,cv=cv,scoring="balanced_accuracy",error_score="raise",n_jobs=1)
        scores[name]={"folds":values.tolist(),"mean":float(values.mean()),"std_descriptive":float(values.std())}
        print(name,scores[name],flush=True)
    search=Pipeline([("preprocessor",clone(preprocessing)),("classifier",rf())])
    grid=[
        {"classifier":[DecisionTreeClassifier(random_state=args.seed)],"classifier__max_depth":[2,4]},
        {"classifier":[rf()],"classifier__n_estimators":[100,300]},
        {"classifier":[LogisticRegression(max_iter=3000,random_state=args.seed)],"classifier__C":[.01,1,100]},
    ]
    tuning=GridSearchCV(search,grid,scoring="balanced_accuracy",cv=cv,n_jobs=1,error_score="raise")
    tuning.fit(x_train,y_train)
    candidates["tuned_grid"]=tuning.best_estimator_
    scores["tuned_grid"]={"mean":float(tuning.best_score_),"selection_score_optimistic":True,
        "best_params":{k:str(v) for k,v in tuning.best_params_.items()}}
    selected=max(scores,key=lambda name:scores[name]["mean"])
    # Freeze selection before computing any test scores.
    candidates["always_negative"]=DummyClassifier(strategy="constant",constant=-1)
    results={}
    predictions=pd.DataFrame({"row_id":test,"profile_group":groups[test],"actual":y_test.to_numpy()})
    for name,model in candidates.items():
        model.fit(x_train,y_train)
        predicted=pd.Series(model.predict(x_test),index=x_test.index)
        positive=model.predict_proba(x_test)[:,list(model.classes_).index(1)]
        results[name]=evaluate(y_test,predicted,positive)
        predictions[name]=predicted.to_numpy()
        predictions[name+"_score"]=positive
    perm=permutation_importance(candidates[selected],x_test,y_test,scoring="balanced_accuracy",n_repeats=5,random_state=args.seed,n_jobs=1)
    conflicts=pd.DataFrame({"g":groups,"y":y}).groupby("g")["y"].nunique()
    artifact={
        "analysis_version":"2026-11-10-reviewed",
        "status":"educational rerun of previously inspected data; not fresh external validation",
        "data_sha256":hashlib.sha256(args.data.read_bytes()).hexdigest(),"data_file":args.data.name,
        "rows":len(x),"features":FEATURES,"labels":LABELS,
        "class_counts":{str(k):int(v) for k,v in y.value_counts().items()},
        "unique_feature_profiles":int(len(np.unique(groups))),
        "repeated_feature_profile_rows":int(len(x)-len(np.unique(groups))),
        "feature_profiles_with_conflicting_labels":int((conflicts>1).sum()),
        "partition":{"method":args.split_strategy,"model_seed":args.seed,
            "split_seed":args.seed if args.split_strategy=="row" else None,
            "cv_seed":100 if args.split_strategy=="row" else None,"cv_splits":args.cv_splits,
            "group_allocation":"deterministic, first outer fold" if args.split_strategy=="profile" else None,
            "train_rows":len(train),"test_rows":len(test),"test_fraction":len(test)/len(x),
            "overlapping_profile_groups":len(set(groups[train])&set(groups[test])),
            "train_positive":int((y_train==1).sum()),"test_positive":int((y_test==1).sum())},
        "selection":{"metric":"training CV balanced accuracy","selected_model":selected,"test_used_for_selection":False},
        "cv":scores,"test_metrics":results,
        "permutation_importance":{"model":selected,"scoring":"balanced_accuracy","repeats":5,
            "features":[{"feature":f,"mean":float(m),"std":float(s)} for f,m,s in zip(FEATURES,perm.importances_mean,perm.importances_std)]},
        "limitations":["Exact profiles are not patient IDs; repeated profiles do not prove duplicate patients.",
            "Profile grouping measures generalization to unseen exact profiles, unlike ordinary row splitting.",
            "Historical dataset/test information was already inspected; this is reproduction, not prospective validation.",
            "CV means used for selection are optimistic; fold SD is not a confidence interval.",
            "Feature semantics/units, patient identity and collection sampling are not established by the local CSV.",
            "No calibrated clinical risk, clinical utility or causal inference is established."],
        "environment":{"python":platform.python_version(),**{p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","imbalanced-learn"]}},
    }
    args.output_json.parent.mkdir(parents=True,exist_ok=True)
    args.output_json.write_text(json.dumps(artifact,indent=2),encoding="utf-8")
    predictions.to_csv(args.output_json.with_suffix(".predictions.csv"),index=False)
    print(json.dumps({"selected_model":selected,"test_metrics":results},indent=2))
    print("Saved",args.output_json)


if __name__=="__main__":
    main()
