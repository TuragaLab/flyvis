import argparse
import os


def replace_in_markdown_files(folder_path, old_path, new_path):
    # Check if the folder_path exists and is a directory
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: {folder_path} does not exist or is not a directory")
        return

    # Check if the folder_path contains markdown files
    markdown_files = [f for f in os.listdir(folder_path) if f.endswith('.md')]
    if not markdown_files:
        print(f"Error: {folder_path} does not contain any markdown files")
        return

    # Walk through all files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if file is markdown
            if file.endswith('.md'):
                file_path = os.path.join(root, file)

                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Replace the path
                new_content = content.replace(old_path, new_path)

                # Write back only if changes were made
                if new_content != content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Updated: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace paths in markdown files')
    parser.add_argument(
        'folder_path', help='Path to the folder containing markdown files'
    )
    parser.add_argument(
        '--old_path',
        help='Old path to replace',
        default="/groups/turaga/home/lappalainenj/FlyVis/private/flyvision",
    )
    parser.add_argument(
        '--new_path', help='New path to replace with', default="../flyvis"
    )
    args = parser.parse_args()

    replace_in_markdown_files(args.folder_path, args.old_path, args.new_path)
