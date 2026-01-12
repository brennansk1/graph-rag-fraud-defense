from faker import Faker
import pandas as pd

fake = Faker()

def generate_users(n=50000):
    users = []
    for _ in range(n):
        user = {
            'name': fake.name(),
            'ssn': fake.ssn(),
            'address': fake.address(),
            'email': fake.email(),
            'phone': fake.phone_number()
        }
        users.append(user)
    return pd.DataFrame(users)

if __name__ == "__main__":
    df = generate_users()
    df.to_csv('data/raw/legitimate_users.csv', index=False)
    print(f"Generated {len(df)} legitimate users.")