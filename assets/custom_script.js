// Wait for element to load via polling
// function addSankeyModalAttrs() {
// 	var element = document.getElementById("sankey-fig");
// 	// Poll the sankey element variable until it has fully loaded, after which we proceed to adding the modal trigger attributes to the nodes
// 	if (!element) {
// 		setTimeout(addSankeyModalAttrs, 500);
// 		return;
// 	} else {
// 		// console.log(element.getAttribute("data-dash-is-loading"))
// 		if (element.getAttribute("data-dash-is-loading")) {
// 			setTimeout(addSankeyModalAttrs, 500);
// 			return;
// 		}
// 		else {
// 			// console.log(element.className)
// 			if (element.className.includes("dash-graph--pending")) {
// 				setTimeout(addSankeyModalAttrs, 500);
// 				return;
// 			}
// 			else {
// 				var nodes = document.querySelectorAll(".node-rect")
// 				for (let i = 0; i < nodes.length; i++) {
// 					nodes[i].setAttribute("data-toggle", "modal")
// 					nodes[i].setAttribute("data-target", "#myModal")
// 					console.log(nodes[i])
// 				}
// 			}
// 		}
// 	}
// }

// addSankeyModalAttrs()

// // Wait for element to load via promise
// function waitForElm(selector) {
// 	// We use promises when we want code to execute only once an asynchronous operation has completed successfully or with an error e.g. a browser loading an element in the DOM in this example here
// 	// We call a function which returns a promise which may be kept (resolved with a value) or broken (rejected with an error).
// 	// The new Promise() constructor creates a promise object. This has 2 key properties. 1) The state of the promise (pending, fulfilled or rejected) and 2. The result (undefined when state is pending, value when resolve(value) is called and error when reject(error) is called)
// 	// Only once the promise is settled with either a resolve call or a reject call will the callback defined in the then method be executed.
// 	return new Promise(resolve => {
// 		if (document.querySelector(selector)) {
// 			return resolve(document.querySelector(selector));
// 		}

// 		const observer = new MutationObserver(mutations => {
// 			if (document.querySelector(selector)) {
// 				resolve(document.querySelector(selector));
// 				observer.disconnect();
// 			}
// 		});

// 		observer.observe(document.body, {
// 			childList: true,
// 			subtree: true
// 		});
// 	});
// }

function waitForElms(selector) {
	// We use promises when we want code to execute only once an asynchronous operation has completed successfully or with an error e.g. a browser loading an element in the DOM in this example here
	// We call a function which returns a promise which may be kept (resolved with a value) or broken (rejected with an error).
	// The new Promise() constructor creates a promise object. This has 2 key properties. 1) The state of the promise (pending, fulfilled or rejected) and 2. The result (undefined when state is pending, value when resolve(value) is called and error when reject(error) is called)
	// Only once the promise is settled with either a resolve call or a reject call will the callback defined in the then method be executed.
	return new Promise(resolve => {
		if (document.querySelectorAll(selector).length > 0) {
			return resolve(document.querySelectorAll(selector));
		}

		const observer = new MutationObserver(mutations => {
			if (document.querySelectorAll(selector).length > 0) {
				resolve(document.querySelectorAll(selector));
				observer.disconnect();
			}
		});

		observer.observe(document.body, {
			childList: true,
			subtree: true
		});
	});
}

var selSankeyNodeLabels = '#sankey-fig .sankey-node'
waitForElms(selSankeyNodeLabels).then((elms) => {
	for (let i = 0; i < elms.length; i++) {
		const nodeRectHeight = elms[i].querySelector('.node-rect').getBBox().height
		let nodeLabel = elms[i].querySelector('.node-label')
		const nodeLabelLineHeight = parseInt(nodeLabel.style.fontSize)
		// const nodeLabelHeight = nodeLabel.clientHeight
		const translateBy = (nodeRectHeight / 2 - nodeLabelLineHeight * 0 - 2).toString()
		nodeLabel.style.transform = "translate(-5px," + translateBy + "px)"
	}
})

// The following is somewhat of a hack. Dash sets the pointer-events css attribute to "all" for the drag layer of the bar chart and "none" for all other layers including the bar layer. This means only the drag layer will register mouse events including click and hover. If we want the cursor to change on hover over just the bars, we can't do this by setting the pointer-events attribute on the bar layer as this seems to stop click events being registered from the drag layer which we need for interactivity. One solution is to utilise mutations in the hover layer when the cursor is over the bars (the creation and removal of hover text) to trigger a change in cursor on the drag layer.
var selHoverLayers = '#bar-fig-settlements-fsp .hoverlayer, #bar-fig-settlements-zonal .hoverlayer, #bar-fig-settlements-fu .hoverlayer, #funnel-las .hoverlayer, #funnel-gla .hoverlayer'
waitForElms(selHoverLayers).then((elms) => {
	for (let i = 0; i < elms.length; i++) {

		const observer = new MutationObserver(mutationList => {
			dragLayer = elms[i].parentElement.parentElement.querySelectorAll(".draglayer")[0]
			if (mutationList[0].addedNodes.length > 0) {
				dragLayer.style.cursor = "pointer"
			} else {
				dragLayer.style.cursor = "default"
			}

		});

		observer.observe(elms[i], {
			childList: true,
			// subtree: true
		});



	}
})



var selAngularGrid = '#spider-fig g.polarsublayer.angular-grid'
waitForElms(selAngularGrid).then((elms) => {
	selSpiderSvg = '#spider-fig > div.js-plotly-plot > div > div > svg:nth-child(1)'

	spiderSvg = document.querySelector(selSpiderSvg)
	spiderSvgWidth = spiderSvg.clientWidth
	xShift = (spiderSvgWidth / 2).toString()
	grid = elms[0]
	grid.style.transformOrigin = xShift + "px 215px"
	numSectors = grid.children.length
	rotateBy = (360 / numSectors / 2).toString()
	grid.style.transform = "rotate(" + rotateBy + "deg)"
	spiderSvgBbox = spiderSvg.getBBox()
	selAngAxis = '#spider-fig g.polarsublayer.angular-axis';
	angAxis = document.querySelector(selAngAxis);
	selMainDrag = '#spider-fig path.maindrag.drag'
	mainDragOrig = document.querySelector(selMainDrag);
	selPolarDefs = '#spider-fig > div.js-plotly-plot > div > div > svg:nth-child(1) > defs'
	polarDefs = document.querySelector(selPolarDefs);
	mainDrag = mainDragOrig.cloneNode(true)
	mainDragPath = mainDrag.getAttribute('d')
	mainDragPathInverted = mainDragPath.replaceAll("0,1,0", "0,1,1")
	mainDrag.setAttribute('d', mainDragPathInverted)
	mainDrag.style.transformOrigin = xShift + "px 215px"
	rotateByLabels = (90 + 360 / numSectors / 2).toString()
	mainDrag.style.transform = "rotate(" + "-" + rotateByLabels + "deg) " + "translate(" + xShift + "px,215px)"
	mainDrag.setAttribute("id", "curve")
	// console.log(mainDrag)
	polarDefs.appendChild(mainDrag)

	initOffset = rotateBy / 360 * 100
	regOffset = initOffset * 2
	for (let i = 0; i < numSectors; i++) {
		sector_index = (i + 1).toString()
		textElem = document.createElementNS("http://www.w3.org/2000/svg", "text")
		textElem.setAttribute("text-anchor", "middle")
		textElem.classList.add("kpilabel");
		if (i < 2) {
			textElem.classList.add("theme1");
		}
		else if (i < 4) {
			textElem.classList.add("theme2");
		}
		else {
			textElem.classList.add("theme3");
		}

		angAxis.appendChild(textElem)
		textPathElem = document.createElementNS("http://www.w3.org/2000/svg", "textPath")
		textPathElem.setAttributeNS('http://www.w3.org/1999/xlink', "xlink:href", "#curve")
		if (i == 0) {
			offset = initOffset
		} else {
			offset = initOffset + regOffset * i
		}
		textPathElem.setAttribute("startOffset", offset.toString() + "%")
		textPathElem.innerHTML = "KPI " + sector_index
		textElem.appendChild(textPathElem)
	}

	// selLegend = '#spider-fig g.legend'
	// legend = document.querySelector(selLegend)
	// legendHeight = legend.getAttribute('height')
	// console.log(legendHeight)
	selPolarLayer = '#spider-fig g.polarlayer'
	polarLayer = document.querySelector(selPolarLayer)
	polarLayer.style.transform = "translate(0px,10px)"

	numGroups = 3
	groupStarts = [0, 2, 4, 6]
	groupClasses = ["theme1", "theme2", "theme3"]
	groupText = ['CUSTOMER SATISFACTION', 'LAEP PIPELINE', 'LOCAL NET ZERO OFFERING']
	for (let i = 0; i < numGroups; i++) {
		textElem = document.createElementNS("http://www.w3.org/2000/svg", "text");
		textElem.setAttribute("text-anchor", "middle");
		textElem.classList.add("themelabel");
		textElem.classList.add(groupClasses[i]);
		angAxis.appendChild(textElem);
		textPathElem = document.createElementNS("http://www.w3.org/2000/svg", "textPath")
		textPathElem.setAttributeNS('http://www.w3.org/1999/xlink', "xlink:href", "#curve")
		offset = ((groupStarts[i] + groupStarts[i + 1]) / 2 / numSectors) * 100
		textPathElem.setAttribute("startOffset", offset.toString() + "%")
		textElem.appendChild(textPathElem)
		textSpanElem = document.createElementNS("http://www.w3.org/2000/svg", "tspan")
		textSpanElem.setAttribute("dy", "-25")
		textPathElem.appendChild(textSpanElem)
		textSpanElem.innerHTML = groupText[i]
	}

	const observer = new MutationObserver(mutationList => {
		console.log(mutationList)
		spiderSvgWidth = spiderSvg.clientWidth
		grid.style.transformOrigin = (spiderSvgWidth / 2).toString() + "px 215px"
		grid.style.transform = "rotate(" + rotateBy + "deg)"
	});

	observer.observe(spiderSvg, {
		attributes: true,
		attributeFilter: ["width"]
	});

})

selCleoBarPoints = '#cleo-tool-bar > div.js-plotly-plot > div > div > svg:nth-child(1) > g.cartesianlayer > g > g.plot > g > g > g'
waitForElms(selCleoBarPoints).then((elms) => {
	selXTicks = '#cleo-tool-bar > div.js-plotly-plot > div > div > svg:nth-child(1) > g.cartesianlayer > g > g.xaxislayer-above'
	XTicks = document.querySelector(selXTicks)
	XLabelParents = XTicks.children
	let LAIndex = 0
	for (let i = 0; i < XLabelParents.length; i++) {
		XLabel = XLabelParents[i].firstChild.innerHTML;
		console.log(XLabel)
		if (XLabel == "Local Government Authorities") {
			LAIndex = i
		}

	}
	console.log(LAIndex)
	console.log(elms[0].children[LAIndex])
	// elms[0].children[LAIndex].firstChild.setAttribute("fill","#34a996")
	elms[0].children[LAIndex].firstChild.style.fill = "#34a996"

}
)

function getMonthFromString(mon) {
	return new Date(Date.parse(mon + " 1, 2023")).getMonth() + 1
}

selToolTip1 = "#date-slider-dip > div > div:nth-child(7) > div > div > div > div.rc-slider-tooltip-inner"
selToolTip2 = "#date-slider-dip > div > div:nth-child(8) > div > div > div > div.rc-slider-tooltip-inner"
waitForElms(selToolTip2).then((elms) => {
	selFirstDateLabel = "#date-slider-dip > div > div.rc-slider-mark > span:nth-child(1)"
	firstDateString = document.querySelector(selFirstDateLabel).innerHTML
	firstDate = new Date(Date.parse(firstDateString))
	console.log(firstDate)

	toolTip1 = document.querySelector(selToolTip1)
	toolTip2 = document.querySelector(selToolTip2)
	const observer = new MutationObserver(mutationList => {
		handle = mutationList[0]['target']
		attributeName = mutationList[0]['attributeName']
		handlePos = parseInt(handle.getAttribute(attributeName))
		selectedDate = new Date(firstDate.getTime() + (handlePos) * 24 * 60 * 60 * 1000);
		year = String(selectedDate.getFullYear()).slice(-2)
		month = selectedDate.toLocaleString('en-UK', { month: 'short' })
		day = String(selectedDate.getDate()).padStart(2, '0')
		if (handle.classList[1].slice(-1) == '1') {
			toolTip = toolTip1
		} else {
			toolTip = toolTip2
		}
		toolTip.innerHTML = day + " " + month + " " + year
	});
	for (let i = 0; i < 2; i++) {
		selHandle = "#date-slider-dip > div > div.rc-slider-handle.rc-slider-handle-" + String(i + 1)
		handle = document.querySelector(selHandle)
		handlePos = parseInt(handle.getAttribute("aria-valuenow"))
		selectedDate = new Date(firstDate.getTime() + (handlePos) * 24 * 60 * 60 * 1000);
		console.log(selectedDate)
		year = String(selectedDate.getFullYear()).slice(-2)
		month = selectedDate.toLocaleString('en-UK', { month: 'short' })
		day = String(selectedDate.getDate()).padStart(2, '0')
		if (i==0) {
			toolTip = toolTip1
		} else {
			toolTip = toolTip2
		}
		toolTip.innerHTML = day + " " + month + " " + year
		observer.observe(handle, {
			attributes: true,
			attributeFilter: ['aria-valuenow']
		});
	}




}
)

// async function getSankeyNodes(selSankey) {
// 	console.log("about to call waitForElms")
// 	var nodes = await waitForElms(selSankey) // note: await unwraps the promise but async will return a promise anyway!
// 	console.log("waitForElms complete")
// 	return nodes
// }

// var selSankeyNodes = '#sankey-fig .node-rect'
// waitForElms(selSankeyNodes).then((elms) => {
// 	console.log(elms)
// })
// sankeyNodes = getSankeyNodes(selSankeyNodes) // Doesn't work as we getSankeyNodes will return a promise as it is an async function. Use then chaining.

// function delay(ms) {
// 	return new Promise(resolve => {
// 		setTimeout(resolve, ms)
// 	})
// }
// // Use event delegation to listen to events associated with dynamically created elements (here the close modal button) by binding the event to a parent element (body) which is guarenteed to be present.
// var selModalClose = '.close-modal'
// $("body").on("click", selModalClose, async function () {
// 	var selModalWindow = "body > div.fade.modal.show"
// 	var selModalBackdrop = "body > div.fade.modal-backdrop.show"
// 	var modalWindow = $(selModalWindow)
// 	var modalBackdrop = $(selModalBackdrop)
// 	transitionTimeModalWindow = parseFloat(modalWindow.css('transition-duration')) * 1000
// 	transitionTimeModalBackdrop = parseFloat(modalBackdrop.css('transition-duration')) * 1000
// 	var transitionTimeEff = Math.max(transitionTimeModalWindow, transitionTimeModalBackdrop)
// 	modalWindow.removeClass("show")
// 	modalBackdrop.removeClass("show")
// 	console.log("transition start")
// 	await delay(transitionTimeEff) // await keyword within async func ensures that code execution is paused until the promise is resolved
// 	console.log("transition end")
// 	// document.getElementById("modal-body").textContent=""
// 	// modalWindowAlt = document.querySelector("body > div.fade.modal")
// 	// modalWindowBackdropAlt = document.querySelector("body > div.fade.modal-backdrop")
// 	// document.body.removeChild(modalWindowAlt)
// 	// document.body.removeChild(modalWindowBackdropAlt)
// 	modalWindow.remove()
// 	modalBackdrop.remove() // Alternative to remove? Remove modal body children first?
// 	$("body").removeAttr("data-rr-ui-modal-open")
// 	$("body").removeClass("modal-open")
// 	document.body.style = ""
// })