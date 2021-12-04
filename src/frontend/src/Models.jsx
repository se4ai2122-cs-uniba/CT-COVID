import React from 'react'
import {
    VStack,
    Button,
    OrderedList,
    ListItem
} from '@chakra-ui/react'

const axios = require('axios').default
const Endpoint = 'http://localhost:5000/'
const ModelsContext = React.createContext([])

function Models() {
    const [models, setModels] = React.useState([])
    const updateModels = async () => {
        await axios({
            url: Endpoint + 'models',
            method: 'get',
            headers: {
                Accept: 'application/json'
            }
        })
        .then(function(response) {
            setModels(response.data.models)
        })
        .catch(function(error) {
            console.log(error)
        })
    }

    return (
        <ModelsContext.Provider value={{models}}>
            <VStack>
                <Button
                    size="sm"
                    colorScheme='teal'
                    variant='solid'
                    onClick={updateModels}>
                    Get Models List
                </Button>
                <OrderedList>
                {
                    models.map((name) => (
                        <ListItem key={name}>{name}</ListItem>
                    ))
                }
                </OrderedList>
            </VStack>
        </ModelsContext.Provider>
    )
}

export default Models
