import React from 'react'
import ReactDOM from 'react-dom'
import {
    ChakraProvider,
    Center,
    VStack,
    Divider
} from '@chakra-ui/react'

import Header from './Header'
import Models from './Models'
import Predict from './Predict'

function App({ Component }) {
  return (
    <ChakraProvider>
      <Header />
      <Center bg='white' padding='5mm'>
          <VStack divider={<Divider />} >
              <Models />
              <br />
              <Predict />
          </VStack>
      </Center>
    </ChakraProvider>
  )
}

const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)
