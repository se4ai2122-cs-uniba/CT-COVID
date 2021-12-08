import {
    Center,
    VStack,
    Divider,
    Heading,
    Text
} from '@chakra-ui/react'

const Header = () => {
    return (
        <Center bg='white' padding='5mm'>
            <VStack divider={<Divider />} >
                <Heading as='h1' size='lg'>CT-COVID</Heading>
                <Text>Screening CT 3d images for interpretable COVID19 detection.</Text>
            </VStack>
        </Center>
    )
}

export default Header
