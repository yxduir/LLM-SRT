import csv
import argparse
from collections import defaultdict

def analyze_comet(csv_file, top_n=None):
    eng_to_x = defaultdict(list)
    x_to_eng = defaultdict(list)
    cmn_to_x = defaultdict(list)
    x_to_cmn = defaultdict(list)

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            iso3 = row['iso3']
            comet = float(row['comet'])
            
            src, tgt = iso3.split('_')
            
            if src == 'eng':
                eng_to_x[tgt].append(comet)
            elif tgt == 'eng':
                x_to_eng[src].append(comet)
            
            if src == 'cmn':
                cmn_to_x[tgt].append(comet)
            elif tgt == 'cmn':
                x_to_cmn[src].append(comet)

    lang_scores = defaultdict(lambda: {"eng→x": [], "x→eng": [], "cmn→x": [], "x→cmn": []})

    for tgt_lang, scores in eng_to_x.items():
        lang_scores[tgt_lang]["eng→x"] = scores

    for src_lang, scores in x_to_eng.items():
        lang_scores[src_lang]["x→eng"] = scores

    for tgt_lang, scores in cmn_to_x.items():
        lang_scores[tgt_lang]["cmn→x"] = scores

    for src_lang, scores in x_to_cmn.items():
        lang_scores[src_lang]["x→cmn"] = scores

    results = []
    for lang, directions in lang_scores.items():
        all_scores = []
        for direction, scores in directions.items():
            all_scores.extend(scores)
        
        if all_scores:
            avg_comet = sum(all_scores) / len(all_scores)
            eng_to_x_avg = sum(directions["eng→x"]) / len(directions["eng→x"]) if directions["eng→x"] else 0
            x_to_eng_avg = sum(directions["x→eng"]) / len(directions["x→eng"]) if directions["x→eng"] else 0
            cmn_to_x_avg = sum(directions["cmn→x"]) / len(directions["cmn→x"]) if directions["cmn→x"] else 0
            x_to_cmn_avg = sum(directions["x→cmn"]) / len(directions["x→cmn"]) if directions["x→cmn"] else 0
            
            results.append({
                "lang": lang,
                "avg_comet": avg_comet,
                "eng→x": eng_to_x_avg,
                "x→eng": x_to_eng_avg,
                "cmn→x": cmn_to_x_avg,
                "x→cmn": x_to_cmn_avg,
                "count": len(all_scores)
            })

    results.sort(key=lambda x: x["avg_comet"], reverse=True)

    print(f"{'语种':<8} {'4方向均分':<10} {'eng→x':<8} {'x→eng':<8} {'cmn→x':<8} {'x→cmn':<8} {'样本数':<6}")
    print("-" * 65)
    
    display_results = results[:top_n] if top_n else results
    
    for r in display_results:
        print(f"{r['lang']:<8} {r['avg_comet']:<10.2f} {r['eng→x']:<8.2f} {r['x→eng']:<8.2f} {r['cmn→x']:<8.2f} {r['x→cmn']:<8.2f} {r['count']:<6}")

    top_langs = [r["lang"] for r in display_results]
    print("\n" + "=" * 65)
    print(f"Top {len(top_langs)} 语种列表: {top_langs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析COMET分数")
    parser.add_argument("--file", type=str, default="/code/LLM-SRT/eval/csv/20260304/srt_test_idx_vlm_mt_LMT-60-4B_evaluated.csv", help="CSV文件路径")
    parser.add_argument("-n", "--top", type=int, default=50, help="输出前N个语种，默认全部")
    args = parser.parse_args()
    
    analyze_comet(args.file, args.top)
