Direction of Arrival with One Microphone, a few LEGOs, and Non-Negative Matrix Factorization
============================================================================================

This repository contains the code to reproduce the results of the paper
[*Direction of Arrival with One Microphone, a few LEGOs, and Non-Negative Matrix Factorization*](http://arxiv.org/abs/1801.03740).


Abstract
--------

Conventional approaches to sound source localization require at least two microphones. It is known, however, that people with unilateral hearing loss can also localize sounds. Monaural localization is possible thanks to the scattering by the head, though it hinges on learning the spectra of the various sources. We take inspiration from this human ability to propose algorithms for accurate sound source localization using a single microphone embedded in an arbitrary scattering structure. The structure modifies the frequency response of the microphone in a direction-dependent way giving each direction a signature. While knowing those signatures is sufficient to localize sources of white noise, localizing speech is much more challenging: it is an ill-posed inverse problem which we regularize by prior knowledge in the form of learned non-negative dictionaries. We demonstrate a monaural speech localization algorithm based on non-negative matrix factorization that does not depend on sophisticated, designed scatterers. In fact, we show experimental results with ad hoc scatterers made of LEGO bricks. Even with these rudimentary structures we can accurately localize arbitrary speakers; that is, we do not need to learn the dictionary for the particular speaker to be localized. Finally, we discuss multi-source localization and the related limitations of our approach.



Authors
-------

Dalia El Badawy and Ivan Dokmanić 


#### Contact

[Dalia El Badawy](mailto:dalia[dot]elbadawy[at]epfl[dot]ch) <br>

LEGO Devices
------------

<img src="https://raw.githubusercontent.com/swing-research/scatsense/master/images/lego.png" width=500>

The devices made from LEGO bricks are placed on a base plate of size 25 cm by 25 cm. Left: LEGO1. Right: LEGO2. 

The impulse responses measured in an anechoic chamber are stored in `data` as numpy arrays sampled at 16000 Hz. Each column corresponds to one direction.


Dependencies
------------

* [Python 2.7](https://www.python.org/downloads/).
* [Numpy](http://www.numpy.org/), [Protocol Buffers](http://developers.google.com/protocol-buffers/), [Seaborn](http://seaborn.pydata.org/), and [Matplotlib](http://matplotlib.org/).


Reproduce the results
--------------------------------------

### Configuration files

Each experiment is goverened by a configuration file specifying which device to use, the spatial discretization, ... etc. The structure of the file and all the parameters are specified in `proto/info.proto`.
The configuration files used in the paper can be found in `configs`.

### Scripts

There are three main scripts:
* exp_white.py: for results with white sources
* exp_proto.py: for results using a dictionary of spectral prototypes
* exp.py: for results using a universal speech model (USM)

Each script takes as input a configuration file. For example, in a UNIX terminal, run the following

    python exp.py configs/lego1_fspeech_1.txt

A log file, figure, and text file with the results are generated in a subdirectory of  `results`.

