# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...file_utils import _BaseLazyModule, is_tf_available, is_tokenizers_available, is_torch_available


_import_structure = {
    "configuration_electra_cb": ["ELECTRACB_PRETRAINED_CONFIG_ARCHIVE_MAP", "ElectraCBConfig"],
    "tokenization_electra_cb": ["ElectraCBTokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_electra_cb_fast"] = ["ElectraCBTokenizerFast"]

if is_torch_available():
    _import_structure["modeling_electra_cb"] = [
        "ELECTRACB_PRETRAINED_MODEL_ARCHIVE_LIST",
        "ElectraCBForMaskedLM",
        "ElectraCBForMultipleChoice",
        "ElectraCBForPreTraining",
        "ElectraCBForQuestionAnswering",
        "ElectraCBForSequenceClassification",
        "ElectraCBForTokenClassification",
        "ElectraCBModel",
        "ElectraCBPreTrainedModel",
        "load_tf_weights_in_electra_cb",
    ]

if is_tf_available():
    _import_structure["modeling_tf_electra_cb"] = [
        "TF_ELECTRACB_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFElectraCBForMaskedLM",
        "TFElectraCBForMultipleChoice",
        "TFElectraCBForPreTraining",
        "TFElectraCBForQuestionAnswering",
        "TFElectraCBForSequenceClassification",
        "TFElectraCBForTokenClassification",
        "TFElectraCBModel",
        "TFElectraCBPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_electra_cb import ELECTRACB_PRETRAINED_CONFIG_ARCHIVE_MAP, ElectraConfig
    from .tokenization_electra_cb import ElectraCBTokenizer

    if is_tokenizers_available():
        from .tokenization_electra_cb_fast import ElectraCBTokenizerFast

    if is_torch_available():
        from .modeling_electra_cb import (
            ELECTRACB_PRETRAINED_MODEL_ARCHIVE_LIST,
            ElectraCBForMaskedLM,
            ElectraCBForMultipleChoice,
            ElectraCBForPreTraining,
            ElectraCBForQuestionAnswering,
            ElectraCBForSequenceClassification,
            ElectraCBForTokenClassification,
            ElectraCBModel,
            ElectraCBPreTrainedModel,
            load_tf_weights_in_electra_cb,
        )

    if is_tf_available():
        from .modeling_tf_electra_cb import (
            TF_ELECTRACB_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFElectraCBForMaskedLM,
            TFElectraCBForMultipleChoice,
            TFElectraCBForPreTraining,
            TFElectraCBForQuestionAnswering,
            TFElectraCBForSequenceClassification,
            TFElectraCBForTokenClassification,
            TFElectraCBModel,
            TFElectraCBPreTrainedModel,
        )

else:
    import importlib
    import os
    import sys

    class _LazyModule(_BaseLazyModule):
        """
        Module class that surfaces all objects but only performs associated imports when the objects are requested.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        def _get_module(self, module_name: str):
            return importlib.import_module("." + module_name, self.__name__)

    sys.modules[__name__] = _LazyModule(__name__, _import_structure)
