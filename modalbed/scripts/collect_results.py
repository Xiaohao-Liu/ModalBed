from fire import Fire
import os
import sys
import json

import numpy as np

from modalbed import datasets as datasetlib
from modalbed import algorithms
from modalbed.lib import misc, reporting
from modalbed import model_selection
from modalbed.lib.query import Q


modalityMap = lambda x:{
        "video": "Vid",
        "vision": "RGB",
        "depth": "Dep",
        "audio": "Aud",
        "text": "Lan"
    }.get(x, x)

def format_latex_name(name, rm=False):
    if rm:
        name = name.replace("_", "-")
    if name.find("_") != -1:
        main_, sub_ = name.split("_")
        name = main_ + "$_\\text{%s}$" % sub_
    name = name.replace("Plus", "+")
    
    if name == "VGGSound":
        name = "VGGSound-S"
    return name

def format_perceptor_name(name, rm=False):
    if rm:
        name = name.split("_strongMG")[0]
    else:
        name = name.lower()
        name = name.replace("_strongmg", "_strongMG")
    return name
    
    
def remove_key(d,key):
    new_d = d.copy()
    new_d.pop(key)
    return new_d

def recursive_freeze(obj):
    if isinstance(obj, dict):
        return frozenset((key, recursive_freeze(val)) for key, val in obj.items())
    elif isinstance(obj, list):
        return tuple(recursive_freeze(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(recursive_freeze(item) for item in obj)
    elif isinstance(obj, tuple):
        return tuple(recursive_freeze(item) for item in obj)
    else:
        return obj


def format_mean(data):
    """Given a list of datapoints, return a string describing their mean and
    standard error"""
    if len(data) == 0:
        return None, None, "\\multicolumn{1}{c}{-}"
    mean = 100 * np.mean(list(data))
    err = 100 * np.std(list(data) / np.sqrt(len(data)))
    return mean, err, "\\multicolumn{1}{c}{$\\text{%.1f}_{\\pm\\text{%.1f}}$}" % (
        mean, err
        )
    
def merge_records(records):
    merged_records = []
    args_set = set()  # Store unique args dictionaries

    # Group records by unique 'args' dictionaries
    for record in records:
        args = record['args'].copy()
        args.pop('holdout_fraction', None)  # Remove 'holdout_fraction' from comparison
        args_key = recursive_freeze(args)
        args_set.add(args_key)

    # Merge records with the same 'args' except for 'holdout_fraction'
    for args_key in args_set:
        args_dict = dict(args_key)
        filtered_records = [record for record in records if dict(recursive_freeze(remove_key(record['args'],'holdout_fraction'))) == args_dict]
        merged_record = {}
        for record in filtered_records:
            merged_record.update(record)
        merged_records.append(merged_record)
    return Q(merged_records)



def print_results_tables(records, R_perceptors, R_modals, R_datasets, selection_method, mode):
    
    grouped_records = reporting.get_grouped_records(records)

    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])
    
    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)
    
    alg_names = Q(records).select("args.algorithm").unique()
    
    
    alg_names_dg = [
        "ERM",
        "IRM",
        "Mixup",
        "CDANN",
        "SagNet",
        "IB_ERM",
        "CondCAD",
        "EQRM",
        "ADRMX",
        "ERMPlusPlus",
        "URM",
        ]
    
    alg_names_dg = ([n for n in algorithms.ALGORITHMS if n in alg_names_dg])
    
    alg_names_mml = ["Concat", "OGM", "LFM"]
    
    alg_name_change = lambda x: {"LFM": "DLMG"}.get(x, x)
            
    dataset_names = Q(records).select("args.dataset").unique().sorted()
    dataset_names = [d for d in datasetlib.DATASETS if d in dataset_names]
    
    perceptor_names = Q(records).select("args.perceptor").unique().sorted()
    
    perceptor_names = [p for p in R_perceptors if format_perceptor_name(p) in perceptor_names]
    
    
    print("""\\multirow{2}{*}{\\textbf{Perceptor}}& \\multicolumn{2}{c|}{\\multirow{2}{*}{\\textbf{Method}}} %s \\\\""" %
              (
                    " ".join(["& \\multicolumn{%d}{c%s}{\\textbf{ %s }} " % (len(R_modals[idx])+1, "|" if idx != len(R_datasets) - 1 else "", format_latex_name(dataset, rm=True)) for idx, dataset in enumerate(R_datasets)])
              )
              )
    print("\cmidrule{4-%d}" % (3 + np.sum([len(i) for i in R_modals]) + len(R_datasets)))
    
    print("& & & %s \\\\" % 
          (
              " & ".join(
                ["%s & \\multicolumn{1}{c%s}{\\textbf{Avg}}"
                    % (
                        " & ".join(
                        ["\\multicolumn{1}{c}{\\textbf{%s}}" % modalityMap(modal) for modal in R_modals[idx]]),
                        "|" if idx != len(R_modals) - 1 else ""
                    )
                    for idx in range(len(R_datasets))
                ]
            )
          )
          )
    print("\\midrule")
    for p_idx, perceptor in enumerate(R_perceptors):
        
        p_prefix = "\\multicolumn{1}{c}{\\multirow{%d}{*}{\\rotatebox{90}{%s}}}" % (len(alg_names_dg)+len(alg_names_mml), format_perceptor_name(perceptor, rm=True) )
        
        for a_idx_s, alg_name_s in enumerate(alg_names_mml):
            line=[]
            for r_idx, dataset in enumerate(R_datasets):
                means = []
                for test_env in range(datasetlib.num_environments(dataset)):
                    sample = grouped_records.filter_equals(
                                    "perceptor, dataset, algorithm, test_env",
                                    (format_perceptor_name(perceptor), dataset, alg_name_s, test_env)
                                )
                    # import pdb; pdb.set_trace()
                    trial_accs = sample.select("sweep_acc")
                    if len(sample)>0:
                        p_name = sample[0]['records'][0]['args'].get('perceptor',"")
                        if p_name != format_perceptor_name(perceptor):
                            trial_accs = []
                    else:
                        trial_accs = []
                    mean, err, d = format_mean(trial_accs)
                    line.append(d)
                    means.append(mean)
                    
                if None in means:
                    line.append("\multicolumn{1}{c%s}{-}" % ("|" if (r_idx < len(R_datasets) -1) else ""))
                else:    
                    line.append(
                        "\multicolumn{1}{c%s}{\\text{%.1f}}" % ("|" if (r_idx < len(R_datasets) -1) else "", sum(means) / len(means))
                                )
                    
            s = " & \\multicolumn{1}{l|}{%s} &%s \\\\" % (format_latex_name(alg_name_change(alg_name_s)), " & ".join(line))
            
            if a_idx_s == 0:
                s = "%s & \\multicolumn{1}{c}{\\multirow{%d}{*}{MML}}" % (p_prefix, len(alg_names_mml)) + s
                if len(p_prefix) > len("\\multicolumn{1}{c}{}"):
                    p_prefix = "\\multicolumn{1}{c}{}"
            else:
                s = "\\multicolumn{1}{c}{} & " + s
            print(s)
        
        print("\cmidrule{2-%d}" % (3 + np.sum([len(i) for i in R_modals]) + len(R_datasets)))
            
        for a_idx_w, alg_name_w in enumerate(alg_names_dg):
            line = []
            for r_idx, dataset in enumerate(R_datasets):
                means = []
                for test_env in range(datasetlib.num_environments(dataset)):
                    sample = grouped_records.filter_equals(
                                    "perceptor, dataset, algorithm, test_env",
                                    (format_perceptor_name(perceptor), dataset, alg_name_w, test_env)
                                )
                    # import pdb; pdb.set_trace()
                    trial_accs = sample.select("sweep_acc")
                    if len(sample)>0:
                        p_name = sample[0]['records'][0]['args'].get('perceptor',"")
                        if p_name != format_perceptor_name(perceptor):
                            trial_accs = []
                    else:
                        trial_accs = []
                    mean, err, d = format_mean(trial_accs)
                    line.append(d)
                    means.append(mean)
                    
                if None in means:
                    line.append("\multicolumn{1}{c%s}{-}" % ("|" if (r_idx < len(R_datasets) -1) else ""))
                else:    
                    line.append(
                        "\multicolumn{1}{c%s}{\\text{%.1f}}" % ("|" if (r_idx < len(R_datasets) -1) else "", sum(means) / len(means))
                                )
            
            s = " & \\multicolumn{1}{l|}{%s} &%s \\\\" % (format_latex_name(alg_name_w), " & ".join(line))
            
            if a_idx_w == 0:
                s = "%s & \\multicolumn{1}{c}{\\multirow{%d}{*}{DG}}"%(p_prefix, len(alg_names_dg)) + s
                if len(p_prefix) > len("\\multicolumn{1}{c}{}"):
                    p_prefix = "\\multicolumn{1}{c}{}"
                
            else:
                s = "\\multicolumn{1}{c}{} & " + s
            print(s)
            
        if p_idx != len(R_perceptors) - 1:
            print("\\midrule")
            
    print("\\bottomrule ")
    
    
def save_results_to_json(records, R_perceptors, R_modals, R_datasets, selection_method, mode, output_file):
    grouped_records = reporting.get_grouped_records(records)

    if selection_method == model_selection.IIDAutoLRAccuracySelectionMethod:
        for r in grouped_records:
            r['records'] = merge_records(r['records'])

    grouped_records = grouped_records.map(lambda group:
        { **group, "sweep_acc": selection_method.sweep_acc(group["records"]) }
    ).filter(lambda g: g["sweep_acc"] is not None)

    results = []
    for perceptor in R_perceptors:
        for alg_name in ["Concat", "OGM", "LFM"] + [
            "ERM", "IRM", "Mixup", "CDANN", "SagNet", "IB_ERM", "CondCAD", "EQRM", "ADRMX", "ERMPlusPlus", "URM"
        ]:
            for dataset in R_datasets:
                means = []
                for test_env in range(datasetlib.num_environments(dataset)):
                    sample = grouped_records.filter_equals(
                        "perceptor, dataset, algorithm, test_env",
                        (format_perceptor_name(perceptor), dataset, alg_name, test_env)
                    )
                    trial_accs = sample.select("sweep_acc")
                    if len(sample) > 0:
                        p_name = sample[0]['records'][0]['args'].get('perceptor', "")
                        if p_name != format_perceptor_name(perceptor):
                            trial_accs = []
                    else:
                        trial_accs = []
                    mean, err, _ = format_mean(trial_accs)
                    means.append(mean)
                    results.append({
                        "perceptor": perceptor,
                        "algorithm": alg_name,
                        "dataset": dataset,
                        # "model_select": selection_method,
                        "modality": datasetlib.get_dataset_class(dataset).ENVIRONMENTS[test_env],
                        "mean_accuracy": mean,
                        "std_error": err
                    })
                if None in means:
                    results.append({
                            "perceptor": perceptor,
                            "algorithm": alg_name,
                            "dataset": dataset,
                            # "model_select": selection_method,
                            "modality": "avg",
                            "mean_accuracy": 0,
                            "std_error": 0
                        })
                else:
                    results.append({
                        "perceptor": perceptor,
                        "algorithm": alg_name,
                        "dataset": dataset,
                        # "model_select": selection_method,
                        "modality": "avg",
                        "mean_accuracy": sum(means) / len(means),
                        "std_error": 0
                    })

    with open(output_file.split(".json")[0] + f"_{selection_method.name}.json", 'w') as f:
        json.dump(results, f, indent=4)

def main(
    mode = "weak"
):
    if mode == "weak":
        input_dir=["msrvtt_imagebind", "vggs_imagebind", "nyud_imagebind", "msrvtt_unibind", "nyud_unibind", "vggs_unibind", "msrvtt_languagebind", "nyud_languagebind", "vggs_languagebind", "msrvtt_mml_imagebind", "msrvtt_mml_languagebind", "msrvtt_mml_unibind","vggs_mml_imagebind", "vggs_mml_languagebind", "vggs_mml_unibind"]
        perceptors = ["ImageBind", "LanguageBind", "UniBind"]
        datasets = ["MSR_VTT", "NYUDv2", "VGGSound"]
        
    elif mode == "strong":
        input_dir=["nyud_imagebind_SMG_ori", "nyud_languagebind_SMG_ori", "nyud_unibind_SMG_ori", "msrvtt_imagebind_SMG", "nyud_imagebind_SMG", "msrvtt_unibind_SMG", "nyud_unibind_SMG", "msrvtt_languagebind_SMG", "nyud_languagebind_SMG", "vggs_languagebind_SMG", "vggs_unibind_SMG", "vggs_imagebind_SMG", "msrvtt_mml_imagebind_SMG", "msrvtt_mml_languagebind_SMG", "msrvtt_mml_unibind_SMG","vggs_mml_imagebind_SMG", "vggs_mml_languagebind_SMG", "vggs_mml_unibind_SMG", "nyud_unibind_SMG_ERM"]
        perceptors = ["ImageBind_strongMG", "LanguageBind_strongMG", "UniBind_strongMG"]
        datasets = ["MSR_VTT", "NYUDv2", "VGGSound"]

    results_file = f"modalbed/results/results_{mode}.tex"
    json_file = f"modalbed/results/results_{mode}.json"
    
    scale_size = 0.8
    
    sys.stdout = misc.Tee(os.path.join("./",results_file), "w")
    
    modals = []
    for i in datasets:
        modals.append(datasetlib.get_dataset_class(i).ENVIRONMENTS)
    
    print("\\begin{table}[!h]")
    print("\\centering")
    print("\\caption{Mean and standard deviation of classification performance comparison under the %s MG setting.}" % mode.capitalize())
    print("\\label{tab:%s_mg}" % mode.lower())
    print("\\resizebox{%.2f\\textwidth}{!}{ %%%%%%%%%% scale table" % scale_size)
    print("\setlength{\\tabcolsep}{5mm}")
    print("\\begin{tabular}{ccc|%s}" % ("|".join(["l"*(len(i)+1) for i in modals])))
    
    records = reporting.load_records(os.path.join("results_storage", input_dir[0]))
    for i_dir in input_dir[1:]:
        record = reporting.load_records(os.path.join("results_storage", i_dir))
        records._list.extend(record._list)
        

    SELECTION_METHODS = [
            model_selection.IIDAccuracySelectionMethod,
            model_selection.LeaveOneOutSelectionMethod,
            model_selection.OracleSelectionMethod,
        ]
    
    for selection_method in SELECTION_METHODS:
        print("\\toprule")
        print("\\multicolumn{%d}{c}{Model selection: %s}\\\\" % (
            3 + np.sum([len(i) for i in modals]) + len(datasets),
            selection_method.name.replace("domain", "modality")
        ))
        print("\\midrule")

        print_results_tables(records, perceptors, modals, datasets, selection_method, mode)
        print("\\\\")
        save_results_to_json(records, perceptors, modals, datasets, selection_method, mode, json_file)
    
    print("\\end{tabular}")
    print("}")
    print("\\end{table}")

if __name__ == "__main__":
    Fire(main)
    
# python -m modalbed.scripts.collect_results --mode=weak