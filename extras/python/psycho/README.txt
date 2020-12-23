> Copyright 2020 Google LLC
>
> Licensed under the Apache License, Version 2.0 (the "License"); you may not
> use this file except in compliance with the License. You may obtain a copy of
> the License at
>
>     https://www.apache.org/licenses/LICENSE-2.0
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
> WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
> License for the specific language governing permissions and limitations under
> the License.


This code runs in a Jupyter notebook on a Macintosh.

Here are the current installation notes:

pip3 install jupyterlab
pip3 install pyserial
pip3 install numpy
pip3 install matplotlib

pip3 install ipywidgets
pip3 install ipython

I used installer from their website: (64bit OSX)
  https://nodejs.org/en/download/


jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension

jupyter-lab
