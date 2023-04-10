def df_iv_woes(df, target):
    """
    Gets the Information Value and Weight of Evidence for all attributes vs target.
    Arguments: 
        df: Main DataFrame
        target: 1 = Events; 0 = Non events
    Returns
    A DataFrame with IV and WOE values group by bins/category and attribute
    """
    def iv_woe(df, col, target):
        """
        Gets an aux df with the IV and WOE group by decil or category For an attribute
        Arguments:
            df: Main DataFrame
            col: Attribute to get IV and WOE
            target: 1  = Events; 0  = Non events
        """
        # Aux vars
        aux_df = df.loc[:,[col, target]]
        aux_df["attribute"] = col
        # Avoid categorical 
        nunique = len(aux_df[col].unique())
        if(nunique < 10):
            aux_df["bin"] = aux_df[col]
        else:
            aux_df["bin"] = pd.qcut(aux_df[col], 10, labels = False, duplicates = 'drop')

        # Group By
        g_ivwoe = aux_df.groupby(["attribute", "bin"]).agg(
            min_value = (col, "min"),
            max_value = (col, "max"),
            events = (col, "count"),
            good = (target, lambda x: (x == 0).sum()),
            bad = (target, "sum")).reset_index()
        # Main results
        g_ivwoe["%_good"] = g_ivwoe["good"] / g_ivwoe["good"].sum()
        g_ivwoe["%_bad"] = g_ivwoe["bad"] / g_ivwoe["bad"].sum()
        g_ivwoe["woe"] = np.log(g_ivwoe["%_bad"] / g_ivwoe["%_good"])
        g_ivwoe["iv"] = (g_ivwoe["%_bad"] - g_ivwoe["%_good"]) * g_ivwoe["woe"]
        return g_ivwoe
    # Get predictive attributes
    attributes = df.drop(target, axis = 1).columns
    # Empty list
    list_iv_woe = []
    # Loop over all attributes to apply inner function
    for attr in attributes:
        locals()["iv_woe" + str(attr)] = iv_woe(df = df, col = attr, target = target)
        list_iv_woe.append(locals()["iv_woe" + str(attr)])
    # Final DF
    report = pd.concat(list_iv_woe)
    return report