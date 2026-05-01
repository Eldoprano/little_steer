import marimo
app = marimo.App()
@app.cell
def _():
    import marimo as mo
    _x = mo.md("hello")
    return _x
if __name__ == "__main__":
    app.run()
