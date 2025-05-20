import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def draw_pie_chart(pos, neg, neu):
    ratio = {"positive": pos, "negative": neg, "neutral": neu}
    labels = list(ratio.keys())
    sizes = [ratio[label] for label in labels]
    colors = ["gold", "yellowgreen", "lightcoral"]
    explode = (0, 0, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%", shadow=True, startangle=90)
    ax.axis("equal")
    return fig
