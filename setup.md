# How to upload to pypi for dummies (me)


check-manifest -u -v

python -m build

twine upload --repository testpypi dist/*



increment the counter every time you mess up :>


### test:

pip install -i https://test.pypi.org/simple/ breaching==0.1.0 # does not install dependencies?

pip install dist/breaching-0.1.1.tar.gz # install distribution directly
