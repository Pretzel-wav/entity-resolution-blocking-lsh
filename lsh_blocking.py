from faker import Faker
import pandas as pd
from simulator import SimulationConfig
import simulator
from datasketch import MinHash
fake = Faker()

# create fake data
fake_data = [{
    'name': fake.name(),
    'address': fake.address(),
    'dob': fake.date_of_birth(),
    'phone': fake.basic_phone_number(),
    'email': fake.email(),
    'vin': fake.vin()
} for _ in range(100)]
df = pd.DataFrame(fake_data)    

# simulate life events to create new rows
sim_configs = [
    SimulationConfig(col='name', generator=fake.name, likelihood=0.1),
    SimulationConfig(col='address', generator=fake.address, likelihood=0.1),
    SimulationConfig(col='dob', generator=fake.date_of_birth, likelihood=0.1),
    SimulationConfig(col='phone', generator=fake.basic_phone_number, likelihood=0.1),
    SimulationConfig(col='email', generator=fake.email, likelihood=0.1),
    SimulationConfig(col='vin', generator=fake.vin, likelihood=0.1)
]
df = simulator.simulate(df, sim_configs, union=True)
df = df.reset_index(drop=True)

# minor data cleaning
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
df['phone'] = df['phone'].str.replace(r'\D', '', regex=True)

# normalize data
for col in df.columns:
    df[col+'_normalized'] = df[col].astype(str).str[:10]

# create blocking key using LSH
def get_minhash(concat_str, num_perm=64):
    m = MinHash(num_perm=num_perm)
    for char in concat_str:
        m.update(char.encode('utf8'))
    hash_values = m.hashvalues
    hash_string = ''.join([str(h) for h in hash_values])
    return hash_string

df['concat'] = df['name_normalized'] + df['address_normalized'] + df['dob_normalized'] + df['phone_normalized'] + df['email_normalized'] + df['vin_normalized']
df['lsh'] = df['concat'].apply(lambda x: get_minhash(x))

print(df.sort_values('lsh').head(30))
df.sort_values('lsh').to_csv('lsh_blocking.csv', index=True)