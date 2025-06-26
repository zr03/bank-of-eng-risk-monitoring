// Get the component functions namespace.
var dagcomponentfuncs = (window.dashAgGridComponentFunctions = window.dashAgGridComponentFunctions || {});


dagcomponentfuncs.DCC_Graph = function (props) {
	const { setData } = props;
	rowId = props.data.id
	if (rowId % 100 == 1) {
		return React.createElement(window.dash_core_components.Graph, {
			figure: props.value,
			style: { height: '100%' },
		});
	}
	else {
		return props.value
	}
};