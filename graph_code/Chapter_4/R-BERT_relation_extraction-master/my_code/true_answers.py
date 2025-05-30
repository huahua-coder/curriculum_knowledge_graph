import csv

def csv_to_txt_with_index(inputcsv, outputtxt):
    # 读取CSV文件
    with open(inputcsv, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # 获取第一列数据，并分割制表符前的内容
        firstcolumn = [row[0].split('\t')[0] for row in reader if row]  # 跳过空行

    # 从8001开始索引
    indexeddata = [(8001 + i, value) for i, value in enumerate(firstcolumn)]

    # 写入TXT文件
    with open(outputtxt, 'w', encoding='utf-8') as txtfile:
        for index, value in indexeddata:
            txtfile.write(f"{index}\t{value}\n")

# 使用示例
inputcsv = 'data-ablation/dev.csv'  # 替换为你的CSV文件路径
outputtxt = 'eval-ablation/true_answers.txt'  # 输出TXT文件路径

csv_to_txt_with_index(inputcsv, outputtxt)
print(f"数据已成功写入 {outputtxt}")


