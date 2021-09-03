# NEASQC lib template

This repository is a template for NEASQC libraries.

## Licence

The `LICENCE` file contains the default licence statement as specified in the proposal and partner agreement.

## Building and installing

For simplicity, an example of `setup.py` file is provided in this template.
Feel free to modify it if you have exotic build recipes.


## Coding conventions

In order to simplify the coding conventions, we provide a pylint.rc file in `misc/pylint.rc`.
This will allow you to easily check your naming conventions and various other aspects.
This is not a strict guidline, as pylint can be quite pedantic sometimes (but also very helpful).

A few remarks:
- pylint can be integrated in most editors (and we strongly advise you to)
- running pylint on several source files in one go can find errors such as circular imports or code duplication:

```bash
python -m pylint --rcfile=./misc/pylint.rc <my_source_dir>
```
or

```bash
pylint --rcfile=./misc/pylint.rc <my_source_dir>
```

depending on how you installed pylint.

## Testing and continuous integration

In order to uniformise the continuous integration process across libraries, we will assume that:
- all the tests related to your library are compatible with pytest
- there exists a 'test' recipe in the `setup.py` file

The default test recipe (in this template) simply calls pytest on the full repository.
Pytest detects:
- any file that starts with test\_ (e.g test\_my\_class.py)
- inside these files, any function that starts test\_
- any class that starts with Test

You can run it with:

```bash
python setup.py test
```

This way, you can write tests either right next to the corresponding code (convenient) or in a `tests` folder at the root of the repository.

If you are not familiar with unit testing and you feel that it's too much for your project, that's fine.
The bare minimum would be to include some run examples wrapped in test functions (functional tests).

Remark that in this template, the same tests are in my\_lib/test\_my\_lib.py and tests/test\_my\_lib.py.

## GitHub CI
This repository contains a GitHub Workflow file that will automatically run pytest when changes are pushed.  
Details on disabling and enabling this feature can be found [here](https://docs.github.com/en/enterprise-server@3.0/actions/managing-workflow-runs/disabling-and-enabling-a-workflow).
