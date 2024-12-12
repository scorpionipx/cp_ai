import os


def fix_yolo_dir(directory):
    """
    Iterates over all .txt files in a directory and modifies numbers
    to have a maximum of 6 decimal places.

    Args:
        directory (str): Path to the directory containing .txt files.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)

                # Read the file
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                # Process each line
                updated_lines = []
                for line in lines:
                    parts = line.split()
                    updated_parts = [f"{float(num):.6f}" if i != 0 else num for i, num in enumerate(parts)]
                    updated_lines.append(" ".join(updated_parts))

                # Write back the modified content
                with open(file_path, 'w') as f:
                    f.write("\n".join(updated_lines) + "\n")

# Example usage:
# fix_yolo_dir("path/to/yolo/directory")
fix_yolo_dir(r'C:\Users\ScorpionIPX\PycharmProjects\cp_ai\dataset_dumitrel\labels\val')