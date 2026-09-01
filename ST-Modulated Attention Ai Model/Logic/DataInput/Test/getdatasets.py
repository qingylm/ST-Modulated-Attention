from datasets import load_from_disk

dataset = load_from_disk("D:\Python_Project\ST-Modulated Attention AI\ST-Modulated Attention Ai Model\data\database\wiki_db")
print("样本数:", len(dataset))
print("字段:", dataset.column_names)
print("第一条样本:", dataset[0])