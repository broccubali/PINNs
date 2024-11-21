import json


def extract_markdown_cells(ipynb_path, output_path=None):
    """
    Extracts markdown cells from a Jupyter notebook.

    Args:
        ipynb_path (str): Path to the input .ipynb file.
        output_path (str, optional): Path to save the extracted markdown cells. If None, prints the markdown content.

    Returns:
        List[str]: List of markdown cell contents.
    """
    # Read the notebook file
    with open(ipynb_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    # Extract markdown cells
    markdown_cells = [
        "".join(cell["source"])
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "markdown"
    ]

    # Save or print the results
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n\n".join(markdown_cells))
    else:
        for i, content in enumerate(markdown_cells, 1):
            print(f"--- Markdown Cell {i} ---\n{content}\n")

    return markdown_cells


# Example usage
input_notebook = "kanv3-0.ipynb"
output_file = "dataset.md"  # Set to None if you don't want to save to file
extract_markdown_cells(input_notebook, output_file)
