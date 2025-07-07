import joblib

def save_obj(obj, filename):
    joblib.dump(obj, filename)
    print(f"Saved object to {filename}")

def load_obj(filename):
    return joblib.load(filename)
