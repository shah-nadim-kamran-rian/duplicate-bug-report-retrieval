import os
import pandas as pd

def read_attribute_file(file_path):
    with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]

def load_bug_folder(folder_path):
    data = {}

    for name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, name)
        if os.path.isfile(file_path):
            data[name.strip()] = pd.Series(read_attribute_file(file_path), dtype="string")

    df = pd.DataFrame(data)
    return df

def load_project(project_path):
    frames = []

    for sub in os.listdir(project_path):
        sub_path = os.path.join(project_path, sub)
        if os.path.isdir(sub_path):
            print("Loading:", sub_path)
            df = load_bug_folder(sub_path)
            df["project_subfolder"] = sub
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)

BASE_PATH = r"C:\Users\ASUS\Desktop\dbd_run\268443\msr2013-bug_dataset-master\msr2013-bug_dataset-master\data"

eclipse = load_project(os.path.join(BASE_PATH, "eclipse", "eclipse"))
mozilla = load_project(os.path.join(BASE_PATH, "mozilla", "mozilla"))

eclipse["dataset_group"] = "eclipse"
mozilla["dataset_group"] = "mozilla"

df = pd.concat([eclipse, mozilla], ignore_index=True, sort=False)

# Cleaning column names 
df.columns = [str(c).strip() for c in df.columns]

print("Columns found:", list(df.columns))

wanted = ["short_desc.xml", "bug_status.xml", "component.xml", "product.xml", "project_subfolder", "dataset_group"]
existing = [c for c in wanted if c in df.columns]
df = df[existing].copy()

rename_map = {
    "short_desc.xml": "text",
    "bug_status.xml": "status",
    "component.xml": "component",
    "product.xml": "product",
}
df = df.rename(columns=rename_map)

print("Columns after rename:", list(df.columns))

if "text" not in df.columns:
    raise ValueError("Column 'short_desc' was not found, so 'text' could not be created.")

# Basic cleaning
df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 10]


df = df.head(10000).copy()

out_path = r"C:\Users\ASUS\Desktop\dbd_run\real_data.csv"
df.to_csv(out_path, index=False, encoding="utf-8")

print("Created:", out_path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
