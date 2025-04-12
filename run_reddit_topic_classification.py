from tweetnlp.util import build_dataset_from_reddit, get_label2id
from tweetnlp.loader import load_trainer
from datasets import concatenate_datasets, DatasetDict, ClassLabel, Features, Value

# Step 1: Load and combine Reddit datasets
ds_pop = build_dataset_from_reddit("datasets/popculturechat_comments.zst", "celebrity_&_pop_culture", max_lines=10000)
ds_biz = build_dataset_from_reddit("datasets/Entrepreneur_comments.zst", "business_&_entrepreneurs", max_lines=10000)
dataset = concatenate_datasets([ds_pop, ds_biz])

# Step 2: Convert string labels to integer class indices
unique_labels = sorted(set(dataset["label"]))  # sorted for consistent label-to-id
class_label = ClassLabel(names=unique_labels)

def encode_label(example):
    example["label"] = class_label.str2int(example["label"])
    return example

dataset = dataset.map(encode_label)

# Step 3: Cast label column to ClassLabel type
dataset = dataset.cast(
    Features({
        "text": Value("string"),
        "label": class_label
    })
)

# Step 4: Wrap in DatasetDict for the trainer
wrapped_dataset = DatasetDict({"train": dataset})

# Step 5: Get label mapping
label2id = get_label2id(wrapped_dataset)

# Step 6: Load and run the trainer
trainer_cls = load_trainer("topic_classification")
print("USING trainer class from:", trainer_cls.__module__)
print("Trainer class object:", trainer_cls)

trainer = trainer_cls(
    language_model="cardiffnlp/twitter-roberta-base",  # Use the base model, not one with a fixed head
    dataset=wrapped_dataset,
    label_to_id=label2id,
    output_dir="./reddit_model"
)

# Step 7: Train the model
trainer.train()