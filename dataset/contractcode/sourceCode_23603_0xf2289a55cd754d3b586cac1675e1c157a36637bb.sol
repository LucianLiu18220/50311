/**
 *Submitted for verification at Etherscan.io on 2016-04-04
*/

contract SimpleStorage {
    uint storedData;
    function set(uint x) {
        storedData = x;
    }
    function get() constant returns (uint retVal) {
        return storedData;
    }
}