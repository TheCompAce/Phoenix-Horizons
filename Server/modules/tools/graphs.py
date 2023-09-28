# modules/tools/graphs.py

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np

def string_to_word_graph(input_string, caption=None, color_palette="viridis", save_path=None):
    """
    Create a word graph from a given string.

    Parameters:
    - input_string (str): The string to create a word graph from.
    - caption (str, optional): The caption for the word graph. Defaults to None.
    - color_palette (str, optional): The color palette to use. Defaults to "viridis".
    - save_path (str, optional): The path to save the word graph image. Defaults to None.
    
    Returns:
    np.array: The image data of the word graph.
    """

    # Generate the word cloud
    wordcloud = WordCloud(
        background_color="white",
        colormap=color_palette,
        max_words=100,
        width=800,
        height=400,
    ).generate(input_string)

    # Get the image data
    image_data = wordcloud.to_array()

    # Save to file if save_path is provided
    if save_path:
        wordcloud.to_file(save_path)

    return image_data

# Example usage
if __name__ == "__main__":
    example_string = "This is an example string for generating a word graph."
    image_data = string_to_word_graph(example_string, caption="Example Word Graph", color_palette="plasma", save_path="example_word_graph.png")
