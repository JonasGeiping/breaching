# How to upload to pypi for dummies (me)


check-manifest -u -v

python -m build

twine upload --repository testpypi dist/*

pip install -i https://test.pypi.org/simple/ breaching==0.1.0
