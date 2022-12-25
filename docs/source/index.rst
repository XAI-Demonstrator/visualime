Welcome to the visuaLIME documentation!
#######################################

*VisuaLIME* is an implementation of *LIME* [1] focused on producing visual local explanations
for image classifiers created as part of the
`XAI Demonstrator project <https://github.com/XAI-Demonstrator/xai-demonstrator>`_.

In contrast to the
`reference implementation <https://github.com/marcotcr/lime>`_, *VisuaLIME*
exclusively supports image classification and gives its users full control over the
properties of the generated explanations.

It was written to produce stable, reliable, and expressive explanations at scale.

[1] Ribeiro et al. (2016): *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.*
`arXiv:1602.04938 <https://arxiv.org/abs/1602.04938>`_

Contents
********

.. toctree::
   :maxdepth: 2

   introduction
   explain
   lime
   feature_selection
   visualize
