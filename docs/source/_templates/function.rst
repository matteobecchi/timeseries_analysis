{{ fullname }}
{{ '=' * fullname | length }}

.. autofunction:: {{ fullname }}

   {{ obj.__doc__ | indent | default('No documentation available.') | strip }}

   :param arg1: Description of arg1.
   :type arg1: Type of arg1
   :param arg2: Description of arg2.
   :type arg2: Type of arg2
   :return: Description of the return value.
   :rtype: Type of the return value