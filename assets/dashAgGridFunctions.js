// Get the component functions namespace.
var dagfuncs = (window.dashAgGridFunctions = window.dashAgGridFunctions || {});

dagfuncs.setRowHeight = function (params) {
	rowId = params.data.id
	if (rowId % 100 === 1) {
		return 400;
	}
	else {
		return 42;
	}
}

dagfuncs.isEditable = function (params) {
	return (params.data.id % 100 === 0)
}

dagfuncs.isBaseRow = function (params) {
	return (params.data.id % 100 === 0)
}

dagfuncs.isAddedRow = function (params) {
	return (params.data.id % 100 > 1)
}

dagfuncs.isSpacerRow = function (params) {
	const rowId = params.data.id
	return ((rowId % 100 === 2) || (rowId % 100 === 99))
}

dagfuncs.isHeaderRow = function (params) {
	const rowId = params.data.id
	return (rowId % 100 === 3)
}

dagfuncs.isGraphRow = function (params) {
	const rowId = params.data.id
	return (rowId % 100 === 1)
}


// dagfuncs.isBottomSpacerRow = function (params) {
// 	const rowId = params.data.id
// 	return (rowId % 100 === 99)
// }

dagfuncs.setColSpanZone = function (params) {
	const rowId = params.data.id
	if ((rowId % 100 === 1) || (rowId % 100 === 2) || (rowId % 100 === 99)) {
		return 100;
	}
	else {
		return 1;
	}
}

dagfuncs.setColSpanPower = function (params) {
	const rowId = params.data.id
	if (rowId % 100 > 2) {
		return 4;
	}
	else {
		return 1;
	}
}

dagfuncs.setColSpanSpend = function (params) {
	const rowId = params.data.id
	if (rowId % 100 > 3) {
		return 3;
	}
	else {
		return 1;
	}
}

dagfuncs.setColSpanEvents = function (params) {
	const rowId = params.data.id
	if (rowId % 100 == 3) {
		return 4;
	}
	else {
		return 1;
	}
}