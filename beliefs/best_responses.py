import pickle
import fire

def load_data(filename):
  """Loads the dictionary from the pkl file."""
  try:
    with open(filename, "rb") as f:
      data = pickle.load(f)
    return data
  except FileNotFoundError:
    print(f"Error: File '{filename}' not found. Creating a new dictionary.")
    return {}

def save_data(data, filename):
  """Saves the dictionary to the pkl file."""
  with open(filename, "wb") as f:
    pickle.dump(data, f)

def display_data(filename):

  data = load_data(filename)
  """Displays the dictionary entries in a user-friendly format."""
  if not data:
    print("Dictionary is empty.")
    return

  st1 = "Scenario"
  offset = max([
    len(name) for name in data
  ]) + 1
  print("-" * 200)
  print(st1 + " " * (offset - len(st1)) + "Best response utility")
  print("-" * 200)
  for key, value in data.items():
    print(f"{key}{' ' * (offset - len(key))}{value}")
  print("-" * 200)

def remove_entry(filename, key):
  data = load_data(filename)
  """Removes an entry from the dictionary based on the key."""
  if key not in data:
    print(f"Key '{key}' not found.")
  else:
    del data[key]
    print(f"Entry with key '{key}' removed.")
    save_data(data, filename)  # Save after removal

def add_entry(filename, key, value):
  data = load_data(filename)
  """Adds a new entry to the dictionary based on user input."""
  data[key] = value
  print(f"Entry with key '{key}' and value '{value}' added.")
  save_data(data, filename)  # Save after addition


if __name__ == '__main__':

    fire.Fire(
        {
            "view": display_data,
            "remove": remove_entry,
            "add": add_entry,
        }
    )