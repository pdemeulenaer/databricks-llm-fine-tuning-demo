import datasets

# train_examples, val_examples = datasets.load_dataset(
#     'math_dataset/arithmetic__mul',
#     split=['train', 'test'],
#     as_supervised=True)

from datasets import load_dataset

# Load a specific configuration
dataset = load_dataset("math-ai/TemplateGSM", "templategsm-1000-1k")

# Access the train split if available
train_dataset = dataset['train']

# print(train_dataset.select[range(3)])

# for i in range(5):
#     print(train_dataset[i])

random_sample = train_dataset.shuffle(seed=422).select(range(5))
for example in random_sample:
    print(example)