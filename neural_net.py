import pandas as pd
import tensorflow as tf
import numpy as np

#To use model, import 'ann_model'; also contains 'mse' method for evaluation

df = pd.read_csv('archive/acs2017_census_tract_data.csv')
df = df.dropna()
df = df[['TotalPop', 'Men', 'Women',
       'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 'Citizen',
       'IncomePerCap',
       'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork']]
train_dataset = df.sample(frac=0.9, random_state=0)
test_dataset = df.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('IncomePerCap')
test_labels = test_features.pop('IncomePerCap')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.asarray(train_features).astype('float32'))

ann_model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(units=28, activation='relu'),
    # tf.keras.layers.Dense(units=14, activation='relu'),
    tf.keras.layers.Dense(units=7, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

ann_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss='mean_squared_error'
    )

history = ann_model.fit(
    train_features,
    train_labels,
    epochs=150,
    batch_size=50,
    validation_split=0.1,
    verbose=2
)


def mse(y, y_prime, ignore_size_mismatch=False):
    if (len(y) != len(y_prime)) and not ignore_size_mismatch:
        print(len(y), len(y_prime))
        raise ValueError("Mismatched lengths")
    sum = 0
    for i in range(min(len(y), len(y_prime))):
        sum += (y[i] - y_prime[i])*(y[i] - y_prime[i])
    return (sum/min(len(y), len(y_prime)))[0]