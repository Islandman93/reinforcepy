class Parameters:
    """
    The purpose of :class:`Parameters` is to create a dictionary of values that can be passed around in a single
    variable. That way all parameters can be set in the main file so they are easy to see. But unlike a dictionary,
    it supports required parameters and optional parameters. It also has validation for each of these functions.

    Parameters
    ----------
    name : str
        The name of these parameters. Currently unused.
    values : dict
        A dictionary of values to initialize with
    """
    def __init__(self, name, values):
        self._name = name
        self.values = values

    def required(self, req_parm_list):
        """
        Required makes sure the parameters passed in are in my local dictionary. If not throws an attribute error

        Parameters
        ----------
        req_parm_list : list
            A list of required parameters as strings

        Raises
        ------
        AttributeError if required parameter is missing
        """
        for req in req_parm_list:
            if req not in self.values:
                raise AttributeError("Required parameter " + req + " missing.")

    def get(self, key):
        """
        Gets a parameter with the given key

        Parameters
        ----------
        key : str
            Parameter name to get

        Raises
        ------
        AttributeError if parameter is not in local dictionary
        """
        if key not in self.values:
            raise AttributeError("Trying to get " + key + " but it doesn't exist. If you're getting this message it's most likely it "
                                           "is not in the required list. Use parms.required(['parm1', 'parm2']) to "
                                           "require parameters")
        else:
            return self.values[key]

    def set(self, key, value):
        """
        Sets a parameters with the given key and value

        Parameters
        ----------
        key : str
            Parameter name
        value : Any
            Parameter value
        """
        self.values[key] = value

    def has(self, key):
        """
        Checks if my local dictionary has the given key

        Parameters
        ----------
        key : str
            Parameter key

        Returns
        -------
        bool
            If key is in parameters dictionary
        """
        return key in self.values

    @staticmethod
    def fromJSON(filename, extra_fns={}):
        import json
        with open(filename, 'r') as in_file:
            in_dict = json.load(in_file)
        return Parameters.fromDict(in_dict, extra_fns)

    @staticmethod
    def fromDict(in_dict, extra_fns={}):
        parameters = dict()

        # loop over file
        for key, parms in in_dict.items():
            # solve common problems with loading
            if key == 'network_parameters':
                # convert "None" to None in input shape
                if 'input_shape' in parms:
                    inp_shape = parms['input_shape']
                    inp_shape = [x if x != "None" else None for x in inp_shape]
                    in_dict[key]['input_shape'] = inp_shape
            if key in extra_fns:
                for param_key, fn in extra_fns[key].items():
                    if param_key in parms:
                        in_dict[key][param_key] = fn(in_dict[key][param_key])

            # end edit add key and params to parameters dictionary
            parameters[key] = Parameters(key, parms)

        return parameters
