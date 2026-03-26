#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Dict, Type

from src.configs.base_modifier_config import ModifierConfig
from src.modifiers.base_modifier import BaseModifier


MODIFIER_MAP: Dict[str, Type[BaseModifier]] = {
}


def get_modifier_from_config(config: ModifierConfig) -> BaseModifier:
    if config.name in MODIFIER_MAP:
        return MODIFIER_MAP[config.name](config)
    else:
        raise NotImplementedError("your modifier is not implemented yet")
