import React from 'react';
import ReactDOM from 'react-dom';
import { ChakraProvider } from '@chakra-ui/react';

import Header from './Header';
import Todos from './Todos'

function App({ Component }) {
  return (
    <ChakraProvider>
      <Header />
      <Todos />
    </ChakraProvider>
  )
}

const rootElement = document.getElementById('root')
ReactDOM.render(<App />, rootElement)
