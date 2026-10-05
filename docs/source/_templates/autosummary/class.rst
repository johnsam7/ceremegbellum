{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% set public_methods = methods | reject("in", ["__init__"]) | reject("match", "_.*") | list %}
   {% if public_methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in public_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}

   {% set public_attributes = attributes | reject("match", "_.*") | list %}
   {% if public_attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in public_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
