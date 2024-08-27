
import bcrypt
import pickle
from pathlib import Path

import streamlit_authenticator as stauth


names = ["Abdoul Aziz Baoula", "Cleeve"]
usernames = ["abdoul_aziz", "cleeve"]
passwords = ["1234", "1234"]

#hashed_passwords = stauth.Hasher(passwords).generate()
# Function to hash passwords
def hash_passwords(passwords):
    hashed = [bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8') for password in passwords]
    return hashed

# Generate hashed passwords
hashed_passwords = hash_passwords(passwords)

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)