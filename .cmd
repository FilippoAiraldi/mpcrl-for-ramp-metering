for /r %%f in (*.py) do (
    pyupgrade --py311-plus "%%f"
)
autoflake --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --in-place --recursive .
isort .
black .
