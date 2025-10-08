from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k")
for i in range(10):
    print(ds["train"][i])

search_str = input("Enter search string: ").strip().lower()

for row in ds["train"]:
    if any(search_str in str(value).lower() for value in row.values()):
        print(row)
