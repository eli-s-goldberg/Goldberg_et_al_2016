TODOs
-----

### Pipeline
- Reorganize output directory structure.
- Figure out wtf to do w/the miscellaneous files (How should they fit into the
  pipeline?).
- Give some thought to the interface presented to users. Presumably they should
  be able to run steps however they want by calling functions, and they should
  be able to run everything together by invoking a binary. It would be nice if
  there was an intermediate run-a-binary-with-options way to get a *little*
  flexibility for people who aren't great coders.
- Figure out how to dump SVGs w/o opening a browser and using crowbar.
- Eli also wants the classifiers to be more pluggable.

### Cleanup and Productionizing
- Break up optimal_tree into functions and test them.
- Many style changes in py files (and their names in some cases).
- Write tests for the helper functions.
- Unify docstring style (to numpy standard).
- Add a license file. BSD? MIT? Apache?
- Reorganize layout to be a releasable package (w/setup.py, etc.)
